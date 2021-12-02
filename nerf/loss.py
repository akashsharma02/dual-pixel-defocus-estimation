"""MPI Optimization functions
"""

import jax.numpy as np
import flax
from jax import jit, lax
import jax.experimental.optimizers
import functools
import multiplane_image
from util import isotropic_total_variation as tv
from util import isotropic_total_variation_batch as tv_batch
import util


def lossfun_helper(params, precomputed_vars, optim_params, patch_params):
    """Compute total loss of the current MPI

  Args:
    params: a dictionary of MPI
    precomputed_vars: a dictionary of precomputed variables
    optim_params: a dictionary of optimization parameters
    patch_params: a dictionary of image patch parameters

  Returns:
    loss: a scalar
    a dictionary of outputs (all-in-focus image & defocus map)
    a dictionary of each loss term
  """

    observations = precomputed_vars["observations"]
    observations_volume = precomputed_vars["observations_volume"]
    filter_halfwidth = precomputed_vars["filter_halfwidth"]
    blur_kernels_scaled = precomputed_vars["blur_kernels_scaled"]
    bias_correction = precomputed_vars["bias_correction"]

    scales = optim_params["scales"]
    weight_loss_data = optim_params["weight_loss_data"]
    weight_loss_aux_data = optim_params["weight_loss_aux_data"]
    weight_prior_sharp_im_tv = optim_params["weight_prior_sharp_im_tv"]
    weight_prior_alpha_tv = optim_params["weight_prior_alpha_tv"]
    weight_prior_entropy = optim_params["weight_prior_entropy"]

    intensity_scale_factor = 0.5 / np.mean(observations)

    # ========== Intensity smoothness prior ==========
    gamma = 1 / 10
    beta = 1 / 32
    sharp_im_tv = tv(np.mean(intensity_scale_factor * sharp_im, axis=-1), gamma)[
        filter_halfwidth:-filter_halfwidth, filter_halfwidth:-filter_halfwidth
    ]
    edge_mask = util.edge_mask_from_image_tv(
        lax.stop_gradient(sharp_im_tv), gamma, beta
    )
    sharp_im_tv_bilateral = sharp_im_tv * (1 - edge_mask)
    sharp_im_tv_per_layer = np.mean(
        (
            scales.size
            * lax.stop_gradient(mpi_transmittance)[..., 0]
            * tv_batch(np.mean(intensity_scale_factor * mpi_colors, axis=-1), gamma)
        )[..., filter_halfwidth:-filter_halfwidth, filter_halfwidth:-filter_halfwidth],
        axis=0,
    )
    edge_mask_per_layer = util.edge_mask_from_image_tv(
        lax.stop_gradient(sharp_im_tv_per_layer), gamma, beta
    )
    sharp_im_tv_per_layer_bilateral = sharp_im_tv_per_layer * (1 - edge_mask_per_layer)
    prior_sharp_im_tv = weight_prior_sharp_im_tv * (
        np.mean(sharp_im_tv_bilateral + sharp_i_tv_per_layer_bilateral)
        + np.mean(sharp_im_tv + sharp_im_tv_per_layer)
    )

    # ========== Alpha and Transmittance smoothness prior ==========
    alpha_tv = (
        tv_batch(np.mean(np.sqrt(mpi_alphas), axis=-1), gamma)[
            ..., filter_halfwidth:-filter_halfwidth, filter_halfwidth:-filter_halfwidth
        ]
        + tv_batch(np.mean(np.sqrt(mpi_transmittance), axis=-1), gamma)[
            ..., filter_halfwidth:-filter_halfwidth, filter_halfwidth:-filter_halfwidth
        ]
    )
    edge_mask_volume = np.repeat(
        edge_mask[None, ...], repeats=alpha_tv.shape[0], axis=0
    )
    alpha_tv_bilateral = alpha_tv * (1 - edge_mask_volume)
    prior_alpha_tv = weight_prior_alpha_tv * (
        np.mean(alpha_tv_bilateral) + np.mean(alpha_tv)
    )

    # ========== Entropy prior ==========
    alpha_entropy = (
        util.collision_entropy(np.sqrt(mpi_alphas[1:, ...]), axis=0)[
            filter_halfwidth:-filter_halfwidth, filter_halfwidth:-filter_halfwidth
        ]
        + util.collision_entropy(np.sqrt(mpi_transmittance[0:, ...]), axis=0)[
            filter_halfwidth:-filter_halfwidth, filter_halfwidth:-filter_halfwidth
        ]
    )
    prior_entropy = weight_prior_entropy * np.mean(alpha_entropy)

    # total loss
    loss = (
        loss_data + loss_aux_data + prior_sharp_im_tv + prior_alpha_tv + prior_entropy
    )

    return (
        loss,
        {
            "sharp_im": sharp_im[
                filter_halfwidth:-filter_halfwidth,
                filter_halfwidth:-filter_halfwidth,
                :,
            ],
            "defocus_map": defocus_map[
                filter_halfwidth:-filter_halfwidth, filter_halfwidth:-filter_halfwidth
            ],
            "mpi": mpi,
        },
        {
            "loss_data": loss_data,
            "loss_aux_data": loss_aux_data,
            "prior_sharp_im_tv": prior_sharp_im_tv,
            "prior_alpha_tv": prior_alpha_tv,
            "prior_entropy": prior_entropy,
        },
    )
