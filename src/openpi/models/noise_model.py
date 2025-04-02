import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at
from typing import TypeVar

ArrayT = TypeVar("ArrayT", at.Array, jax.ShapeDtypeStruct)

NOISE_MODEL_STEP = 100

def sample_noise(
        step: int,
        # action: at.Float[ArrayT, "*b ah ad"],
):
    '''
    Sample noise from a Gaussian distribution with mean 0 and standard deviation 1.
    '''
    # noise = at.Float[ArrayT, "*b ah ad"]
    # noise = jax.random.normal(jax.random.PRNGKey(step), action.shape, dtype=action.dtype)
    if step % NOISE_MODEL_STEP == 0:
        print("noise inserted!")
        print("step", step)
