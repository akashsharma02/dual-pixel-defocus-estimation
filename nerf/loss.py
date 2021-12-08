"""MPI Optimization functions
"""

import jax.numpy as jnp
import flax
from jax import jit, lax
import jax.experimental.optimizers
import functools

def smooth_l1_loss(rendered_pixels, beta=1.0, reduction='mean'):
    """Computes the smooth L1 loss between pair of adjacent pixels.

    Args:
      rendered_pixels: pixel values of shape (n,)
      beta: value of the beta parameter
      reduction: either 'mean' or 'sum' reduction

    Returns:
      loss: sum of smoothness loss between pixels, ni and ni+1
    """
    rendered_pixels = rendered_pixels.reshape((-1, 2))
    xn = rendered_pixels[..., 0]
    yn = rendered_pixels[..., 1]
    smoothness_loss = jnp.where(
            jnp.abs(xn - yn) < beta,
            0.5 * (xn - yn) ** 2 / beta,
            jnp.abs(xn - yn) - 0.5 * beta
        )
    if reduction == 'mean':
        smoothness_loss = jnp.mean(smoothness_loss)
    elif reduction == 'sum':
        smoothness_loss = jnp.sum(smoothness_loss)
    return smoothness_loss
