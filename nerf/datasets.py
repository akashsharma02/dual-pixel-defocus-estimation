# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Different datasets implementation plus a general port for all the datasets."""
from nerf import utils
from PIL import Image
import numpy as np
import jax
import threading
import queue
from pathlib import Path
from os import path
import os
import json
import math

INTERNAL = False  # pylint: disable=g-statement-before-imports
if not INTERNAL:
    import cv2  # pylint: disable=g-import-not-at-top


def get_dataset(split, args):
    return dataset_dict[args.dataset](split, args)


def convert_to_ndc(origins, directions, focal, w, h, near=1.0):
    """Convert a set of rays to NDC coordinates."""
    # Shift ray origins to near plane
    t = -(near + origins[Ellipsis, 2]) / directions[Ellipsis, 2]
    origins = origins + t[Ellipsis, None] * directions

    dx, dy, dz = tuple(np.moveaxis(directions, -1, 0))
    ox, oy, oz = tuple(np.moveaxis(origins, -1, 0))

    # Projection
    o0 = -((2 * focal) / w) * (ox / oz)
    o1 = -((2 * focal) / h) * (oy / oz)
    o2 = 1 + 2 * near / oz

    d0 = -((2 * focal) / w) * (dx / dz - ox / oz)
    d1 = -((2 * focal) / h) * (dy / dz - oy / oz)
    d2 = -2 * near / oz

    origins = np.stack([o0, o1, o2], -1)
    directions = np.stack([d0, d1, d2], -1)
    return origins, directions


class Dataset(threading.Thread):
    """Dataset Base Class."""

    def __init__(self, split, args):
        super(Dataset, self).__init__()
        self.queue = queue.Queue(3)  # Set prefetch buffer to 3 batches.
        self.daemon = True
        self.use_pixel_centers = args.use_pixel_centers
        self.split = split
        if split == "train":
            self._train_init(args)
        elif split == "test":
            self._test_init(args)
        else:
            raise ValueError(
                'the split argument should be either "train" or "test", set'
                "to {} here.".format(split)
            )
        self.batch_size = args.batch_size // jax.process_count()
        self.batching = args.batching
        self.render_path = args.render_path
        self.start()

    def __iter__(self):
        return self

    def __next__(self):
        """Get the next training batch or test example.

        Returns:
          batch: dict, has "pixels" and "rays".
        """
        x = self.queue.get()
        if self.split == "train":
            return utils.shard(x)
        else:
            return utils.to_device(x)

    def peek(self):
        """Peek at the next training batch or test example without dequeuing it.

        Returns:
          batch: dict, has "pixels" and "rays".
        """
        x = self.queue.queue[0].copy()  # Make a copy of the front of the queue.
        if self.split == "train":
            return utils.shard(x)
        else:
            return utils.to_device(x)

    def run(self):
        if self.split == "train":
            next_func = self._next_train
        else:
            next_func = self._next_test
        while True:
            self.queue.put(next_func())

    @property
    def size(self):
        return self.n_examples

    def _train_init(self, args):
        """Initialize training."""
        self._load_renderings(args)
        self._generate_rays()

        if args.batching == "all_images":
            # flatten the ray and image dimension together.
            self.images = self.images.reshape([-1, 3])
            self.rays = utils.namedtuple_map(
                lambda r: r.reshape([-1, r.shape[-1]]), self.rays
            )
        elif args.batching == "single_image":
            self.images = self.images.reshape([-1, self.resolution, 3])
            self.rays = utils.namedtuple_map(
                lambda r: r.reshape([-1, self.resolution, r.shape[-1]]), self.rays
            )
        else:
            raise NotImplementedError(
                f"{args.batching} batching strategy is not implemented."
            )

    def _test_init(self, args):
        self._load_renderings(args)
        self._generate_rays()
        self.it = 0

    def _next_train(self):
        """Sample next training batch."""

        if self.batching == "all_images":
            ray_indices = np.random.randint(
                0, self.rays[0].shape[0], (self.batch_size,)
            )
            batch_pixels = self.images[ray_indices]
            batch_rays = utils.namedtuple_map(lambda r: r[ray_indices], self.rays)
        elif self.batching == "single_image":
            image_index = np.random.randint(0, self.n_examples, ())
            ray_indices = np.random.randint(
                0, self.rays[0][0].shape[0], (self.batch_size,)
            )
            batch_pixels = self.images[image_index][ray_indices]
            batch_rays = utils.namedtuple_map(
                lambda r: r[image_index][ray_indices], self.rays
            )
        else:
            raise NotImplementedError(
                f"{self.batching} batching strategy is not implemented."
            )
        return {"pixels": batch_pixels, "rays": batch_rays}

    def _next_test(self):
        """Sample next test example."""
        idx = self.it
        self.it = (self.it + 1) % self.n_examples

        if self.render_path:
            return {"rays": utils.namedtuple_map(lambda r: r[idx], self.render_rays)}
        else:
            return {
                "pixels": self.images[idx],
                "rays": utils.namedtuple_map(lambda r: r[idx], self.rays),
            }

    # TODO(bydeng): Swap this function with a more flexible camera model.
    def _generate_rays(self):
        """Generating rays for all images."""
        pixel_center = 0.5 if self.use_pixel_centers else 0.0
        x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
            np.arange(self.w, dtype=np.float32) + pixel_center,  # X-Axis (columns)
            np.arange(self.h, dtype=np.float32) + pixel_center,  # Y-Axis (rows)
            indexing="xy",
        )
        camera_dirs = np.stack(
            [
                (x - self.w * 0.5) / self.focal,
                -(y - self.h * 0.5) / self.focal,
                -np.ones_like(x),
            ],
            axis=-1,
        )
        directions = (
            camera_dirs[None, Ellipsis, None, :]
            * self.camtoworlds[:, None, None, :3, :3]
        ).sum(axis=-1)
        origins = np.broadcast_to(
            self.camtoworlds[:, None, None, :3, -1], directions.shape
        )
        viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
        self.rays = utils.Rays(
            origins=origins, directions=directions, viewdirs=viewdirs
        )


class Blender(Dataset):
    """Blender Dataset."""

    def _load_renderings(self, args):
        """Load images from disk."""
        if args.render_path:
            raise ValueError("render_path cannot be used for the blender dataset.")
        with utils.open_file(
            path.join(args.data_dir, "transforms_{}.json".format(self.split)), "r"
        ) as fp:
            meta = json.load(fp)
        images = []
        cams = []
        for i in range(len(meta["frames"])):
            frame = meta["frames"][i]
            fname = os.path.join(args.data_dir, frame["file_path"] + ".png")
            with utils.open_file(fname, "rb") as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.0
                if args.factor == 2:
                    [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
                    image = cv2.resize(
                        image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA
                    )
                elif args.factor > 0:
                    print(args.factor)
                    raise ValueError(
                        "Blender dataset only supports factor=0 or 2, {} "
                        "set.".format(args.factor)
                    )
            cams.append(np.array(frame["transform_matrix"], dtype=np.float32))
            images.append(image)
        self.images = np.stack(images, axis=0)
        if args.white_bkgd:
            self.images = self.images[Ellipsis, :3] * self.images[Ellipsis, -1:] + (
                1.0 - self.images[Ellipsis, -1:]
            )
        else:
            self.images = self.images[Ellipsis, :3]
        self.h, self.w = self.images.shape[1:3]
        self.resolution = self.h * self.w
        self.camtoworlds = np.stack(cams, axis=0)
        camera_angle_x = float(meta["camera_angle_x"])
        self.focal = 0.5 * self.w / np.tan(0.5 * camera_angle_x)
        self.n_examples = self.images.shape[0]


class LLFF(Dataset):
    """LLFF Dataset."""

    def _load_renderings(self, args):
        """Load images from disk."""
        # Load images.
        imgdir_suffix = ""
        if args.factor > 0:
            imgdir_suffix = "_{}".format(args.factor)
            factor = args.factor
        else:
            factor = 1
        imgdir = path.join(args.data_dir, "images" + imgdir_suffix)
        if not utils.file_exists(imgdir):
            raise ValueError("Image folder {} doesn't exist.".format(imgdir))
        imgfiles = [
            path.join(imgdir, f)
            for f in sorted(utils.listdir(imgdir))
            if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
        ]
        images = []
        for imgfile in imgfiles:
            with utils.open_file(imgfile, "rb") as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.0
                images.append(image)
        images = np.stack(images, axis=-1)

        # Load poses and bds.
        with utils.open_file(path.join(args.data_dir, "poses_bounds.npy"), "rb") as fp:
            poses_arr = np.load(fp)
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        bds = poses_arr[:, -2:].transpose([1, 0])
        if poses.shape[-1] != images.shape[-1]:
            raise RuntimeError(
                "Mismatch between imgs {} and poses {}".format(
                    images.shape[-1], poses.shape[-1]
                )
            )

        # Update poses according to downsampling.
        poses[:2, 4, :] = np.array(images.shape[:2]).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] * 1.0 / factor

        # Correct rotation matrix ordering and move variable dim to axis 0.
        poses = np.concatenate(
            [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1
        )
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        images = np.moveaxis(images, -1, 0)
        bds = np.moveaxis(bds, -1, 0).astype(np.float32)

        # Rescale according to a default bd factor.
        scale = 1.0 / (bds.min() * 0.75)
        poses[:, :3, 3] *= scale
        bds *= scale

        # Recenter poses.
        poses = self._recenter_poses(poses)

        # Generate a spiral/spherical ray path for rendering videos.
        if args.spherify:
            poses = self._generate_spherical_poses(poses, bds)
            self.spherify = True
        else:
            self.spherify = False
        if not args.spherify and self.split == "test":
            self._generate_spiral_poses(poses, bds)

        # Select the split.
        i_test = np.arange(images.shape[0])[:: args.llffhold]
        i_train = np.array(
            [i for i in np.arange(int(images.shape[0])) if i not in i_test]
        )
        if self.split == "train":
            indices = i_train
        else:
            indices = i_test
        images = images[indices]
        poses = poses[indices]

        self.images = images
        self.camtoworlds = poses[:, :3, :4]
        self.focal = poses[0, -1, -1]
        self.h, self.w = images.shape[1:3]
        self.resolution = self.h * self.w
        if args.render_path:
            self.n_examples = self.render_poses.shape[0]
        else:
            self.n_examples = images.shape[0]

    def _generate_rays(self):
        """Generate normalized device coordinate rays for llff."""
        if self.split == "test":
            n_render_poses = self.render_poses.shape[0]
            self.camtoworlds = np.concatenate(
                [self.render_poses, self.camtoworlds], axis=0
            )

        super()._generate_rays()

        if not self.spherify:
            ndc_origins, ndc_directions = convert_to_ndc(
                self.rays.origins, self.rays.directions, self.focal, self.w, self.h
            )
            self.rays = utils.Rays(
                origins=ndc_origins,
                directions=ndc_directions,
                viewdirs=self.rays.viewdirs,
            )

        # Split poses from the dataset and generated poses
        if self.split == "test":
            self.camtoworlds = self.camtoworlds[n_render_poses:]
            split = [np.split(r, [n_render_poses], 0) for r in self.rays]
            split0, split1 = zip(*split)
            self.render_rays = utils.Rays(*split0)
            self.rays = utils.Rays(*split1)

    def _recenter_poses(self, poses):
        """Recenter poses according to the original NeRF code."""
        poses_ = poses.copy()
        bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
        c2w = self._poses_avg(poses)
        c2w = np.concatenate([c2w[:3, :4], bottom], -2)
        bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
        poses = np.concatenate([poses[:, :3, :4], bottom], -2)
        poses = np.linalg.inv(c2w) @ poses
        poses_[:, :3, :4] = poses[:, :3, :4]
        poses = poses_
        return poses

    def _poses_avg(self, poses):
        """Average poses according to the original NeRF code."""
        hwf = poses[0, :3, -1:]
        center = poses[:, :3, 3].mean(0)
        vec2 = self._normalize(poses[:, :3, 2].sum(0))
        up = poses[:, :3, 1].sum(0)
        c2w = np.concatenate([self._viewmatrix(vec2, up, center), hwf], 1)
        return c2w

    def _viewmatrix(self, z, up, pos):
        """Construct lookat view matrix."""
        vec2 = self._normalize(z)
        vec1_avg = up
        vec0 = self._normalize(np.cross(vec1_avg, vec2))
        vec1 = self._normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, pos], 1)
        return m

    def _normalize(self, x):
        """Normalization helper function."""
        return x / np.linalg.norm(x)

    def _generate_spiral_poses(self, poses, bds):
        """Generate a spiral path for rendering."""
        c2w = self._poses_avg(poses)
        # Get average pose.
        up = self._normalize(poses[:, :3, 1].sum(0))
        # Find a reasonable "focus depth" for this dataset.
        close_depth, inf_depth = bds.min() * 0.9, bds.max() * 5.0
        dt = 0.75
        mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
        focal = mean_dz
        # Get radii for spiral path.
        tt = poses[:, :3, 3]
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        n_views = 120
        n_rots = 2
        # Generate poses for spiral path.
        render_poses = []
        rads = np.array(list(rads) + [1.0])
        hwf = c2w_path[:, 4:5]
        zrate = 0.5
        for theta in np.linspace(0.0, 2.0 * np.pi * n_rots, n_views + 1)[:-1]:
            c = np.dot(
                c2w[:3, :4],
                (
                    np.array(
                        [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]
                    )
                    * rads
                ),
            )
            z = self._normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
            render_poses.append(np.concatenate([self._viewmatrix(z, up, c), hwf], 1))
        self.render_poses = np.array(render_poses).astype(np.float32)[:, :3, :4]

    def _generate_spherical_poses(self, poses, bds):
        """Generate a 360 degree spherical path for rendering."""
        # pylint: disable=g-long-lambda
        def p34_to_44(p):
            return np.concatenate(
                [
                    p,
                    np.tile(
                        np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1]
                    ),
                ],
                1,
            )

        rays_d = poses[:, :3, 2:3]
        rays_o = poses[:, :3, 3:4]

        def min_line_dist(rays_o, rays_d):
            a_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
            b_i = -a_i @ rays_o
            pt_mindist = np.squeeze(
                -np.linalg.inv((np.transpose(a_i, [0, 2, 1]) @ a_i).mean(0))
                @ (b_i).mean(0)
            )
            return pt_mindist

        pt_mindist = min_line_dist(rays_o, rays_d)
        center = pt_mindist
        up = (poses[:, :3, 3] - center).mean(0)
        vec0 = self._normalize(up)
        vec1 = self._normalize(np.cross([0.1, 0.2, 0.3], vec0))
        vec2 = self._normalize(np.cross(vec0, vec1))
        pos = center
        c2w = np.stack([vec1, vec2, vec0, pos], 1)
        poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])
        rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))
        sc = 1.0 / rad
        poses_reset[:, :3, 3] *= sc
        bds *= sc
        rad *= sc
        centroid = np.mean(poses_reset[:, :3, 3], 0)
        zh = centroid[2]
        radcircle = np.sqrt(rad ** 2 - zh ** 2)
        new_poses = []

        for th in np.linspace(0.0, 2.0 * np.pi, 120):
            camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
            up = np.array([0, 0, -1.0])
            vec2 = self._normalize(camorigin)
            vec0 = self._normalize(np.cross(vec2, up))
            vec1 = self._normalize(np.cross(vec2, vec0))
            pos = camorigin
            p = np.stack([vec0, vec1, vec2, pos], 1)
            new_poses.append(p)

        new_poses = np.stack(new_poses, 0)
        new_poses = np.concatenate(
            [
                new_poses,
                np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape),
            ],
            -1,
        )
        poses_reset = np.concatenate(
            [
                poses_reset[:, :3, :4],
                np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape),
            ],
            -1,
        )
        if self.split == "test":
            self.render_poses = new_poses[:, :3, :4]
        return poses_reset


class Pixel4DP(Dataset):
    """Dataset for Pixel 4 DP data."""

    def _train_init(self, args):
        """Initialize training."""
        self._load_renderings(args)
        self._generate_rays()

        if args.batching == "single_image":
            self.images = self.images.reshape([-1, self.resolution, 2])
            self.rays = utils.namedtuple_map(
                lambda r: r.reshape(
                    [self.resolution, r.shape[-3], r.shape[-2], r.shape[-1]]
                ),
                self.rays,
            )
            self.rays = utils.namedtuple_map(lambda r: np.expand_dims(r, 0), self.rays,)
        else:
            raise NotImplementedError(
                f"{args.batching} batching strategy is not implemented."
            )

    def _next_train(self):
        """Sample next training batch."""

        if self.batching == "single_image":
            # print("rays 0 0 shape 0", self.rays[0][0].shape[0])
            ray_indices = np.random.randint(
                0, self.rays[0][0].shape[0], (self.batch_size,)
            )
            batch_pixels = self.images[0][ray_indices]
            batch_rays = utils.namedtuple_map(lambda r: r[0][ray_indices], self.rays)
        else:
            raise NotImplementedError(
                f"{self.batching} batching strategy is not implemented."
            )
        return {"pixels": batch_pixels, "rays": batch_rays}

    def _next_test(self):
        """Sample next test example."""
        idx = self.it
        self.it = (self.it + 1) % self.n_examples

        return {
            "pixels": self.images[idx],
            "rays": self.rays,
        }

    def _load_renderings(self, args):
        """Load images from disk."""
        if args.render_path:
            raise ValueError("render_path cannot be used for the blender dataset.")

        datapath = Path(args.data_dir)
        meta = None
        with open(datapath / "intrinsics.json") as fp:
            meta = json.load(fp)
        image_fnames = [
            datapath / Path(args.scene + "_left.png"),
            datapath / Path(args.scene + "_right.png"),
        ]

        def _load_and_preprocess_pixel_data(path_to_file):
            # first deduct black level (1024 for 14-bit Google Pixel 4 DP data), then normalize to [0, 1]
            with Image.open(path_to_file) as f:
                image = np.array(f) - 1024
                image[image < 0] = 0
                image = np.stack([np.float32(image)] * 1, axis=2) / (2 ** 14 - 1)
            return image

        # Keep only central field of view (1008 * 1344)
        def _crop_image_central_fov(images, patch_size, num_rows, num_cols):
            """ Crop images

            Args:
              images: [..., H, W, C] #images, height, width, #channels.

            Returns:
              [..., #rows * P, #cols * P, C] cropped images
            """

            crop_y = patch_size * num_rows
            crop_x = patch_size * num_cols
            offset_y = (images.shape[-3] - crop_y) // 2
            offset_x = (images.shape[-2] - crop_x) // 2

            return images[
                ..., offset_y : offset_y + crop_y, offset_x : offset_x + crop_x, :
            ]

        if args.factor > 0:
            print(args.factor)
            raise ValueError(
                "Pixel4DP dataset only supports factor=0 {} " "set.".format(args.factor)
            )

        left_image, right_image = None, None
        left_image = _load_and_preprocess_pixel_data(image_fnames[0])
        right_image = _load_and_preprocess_pixel_data(image_fnames[1])

        patch_params = dict(patch_size=168, num_rows=6, num_cols=8)
        # patch_params = dict(patch_size=21, num_rows=3, num_cols=4)

        self.images = np.stack((left_image, right_image), axis=-1).squeeze()
        self.images = _crop_image_central_fov(self.images, **patch_params)
        self.images = self.images[None, :]
        # print(self.images.shape)

        # TODO: Crop images if we use the calibrated blur kernels
        self.h, self.w = self.images.shape[-3], self.images.shape[-2]
        # print(self.h, self.w)
        self.resolution = self.h * self.w
        self.camera_angle_x = meta["camera_angle_x"]
        # Focal length in pixels (about 801 pixels)
        self.focal = 0.5 * self.w / np.tan(0.5 * self.camera_angle_x)
        # Lens aperture size in m:
        self.aperture_size = (
            math.tan(0.5 * self.camera_angle_x) * self.focal * meta["pixel_pitch"] * 2
        )
        # print(f"Aperture size: {self.aperture_size}")
        self.n_examples = self.images.shape[0]

    def _generate_rays(self):
        """Generating rays for all images."""
        pixel_center = 0.5 if self.use_pixel_centers else 0.0
        x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
            np.arange(self.w, dtype=np.float32) + pixel_center,  # X-Axis (columns)
            np.arange(self.h, dtype=np.float32) + pixel_center,  # Y-Axis (rows)
            indexing="xy",
        )

        camera_dirs = np.stack(
            [
                (x - self.w * 0.5) / self.focal,
                -(y - self.h * 0.5) / self.focal,
                -np.ones_like(x),
            ],
            axis=-1,
        )

        translation_vals = np.linspace(
            -self.aperture_size / 2, self.aperture_size / 2, 8
        )
        translation_x, translation_y = np.meshgrid(
            translation_vals, translation_vals, indexing="xy"
        )
        translations = np.stack(
            [translation_x, translation_y, np.zeros_like(translation_x)], axis=-1
        )

        # print(translations.shape)
        # H x W x 8 x 8 x 3
        directions = np.repeat(
            camera_dirs[..., None, :], len(translation_vals), axis=-2
        )
        # print("shape of directions", directions.shape)
        directions = np.repeat(directions[..., None, :], len(translation_vals), axis=-2)
        # print("shape of directions again", directions.shape)

        origins = np.broadcast_to(translations[None, None, ...], directions.shape)
        # print("shape of oriigins", origins.shape)
        viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
        self.rays = utils.Rays(
            origins=origins, directions=directions, viewdirs=viewdirs
        )


dataset_dict = {
    "blender": Blender,
    "llff": LLFF,
    "Pixel4DP": Pixel4DP,
}
