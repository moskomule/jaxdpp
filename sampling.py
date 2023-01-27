import functools

import jax
from jax import numpy as jnp
from jaxtyping import Array


def _gram_schmidt(vectors: Array
                  ) -> Array:
    # reorthogonalize vectors \R^{DxN} -> \R^{DxN}
    # by modified gram-schmidt
    # input vectors are expected to be linearly independent (except for zero vectors)

    n = vectors.shape[-1]

    def f(vecs, i):
        u = vecs[:, i]
        u = (u / jnp.linalg.norm(u, 2))[:, None]
        weight = jnp.einsum('dm,dn->n', u, vecs)  # proj to u
        masked_weight = jnp.where(jnp.arange(n) > i, weight, 0.0)[None]
        vecs = vecs - jnp.nan_to_num(u * masked_weight)
        return vecs, None

    _vectors, _ = jax.lax.scan(f, vectors, jnp.arange(n))
    return jnp.nan_to_num(_vectors / jnp.linalg.norm(_vectors, ord=2, axis=0, keepdims=True))


def _orthogonal(vectors: Array,
                i: int | Array
                ) -> Array:
    # vectors: \R^{DxN}
    # compute orthogonal basis to e_i

    j = jnp.argmax(jnp.abs(vectors[i]))
    result = vectors - jnp.outer(vectors[:, j] / vectors[i, j], vectors[i])
    # rotate the zero column to the end
    d, n = vectors.shape
    # [0, 1, 2, 3, 4]
    shift_indices = jnp.arange(n)
    # if j == 2: [0, 1, 3, 4, 5]
    shift_indices = shift_indices + (shift_indices >= j)
    # [0, 1, 3, 4, 2]
    shift_indices = jnp.where(shift_indices >= n, j, shift_indices)
    result = result[:, shift_indices]
    result = jnp.where((jnp.arange(d) != i)[:, None] & (jnp.arange(n) != (n - 1)), result, 0)
    return _gram_schmidt(result)


def _sample_edpp(eigen_vecs: Array,
                 edpp_indices: Array,
                 key: jax.random.PRNGKeyArray
                 ) -> Array:
    # eigen_vecs: d x n
    # edpp_indices: n
    # returns a many-hot vector in bool

    d, n = eigen_vecs.shape
    # sort the 1's to the front
    idx = jnp.argsort(~edpp_indices, axis=-1)
    edpp_indices = edpp_indices[idx]
    eigen_vecs = eigen_vecs[:, idx]
    eigen_vecs = eigen_vecs * edpp_indices[None]
    size = edpp_indices.sum()

    def _body(val):
        i, vecs, samples, key = val
        key, key0 = jax.random.split(key)
        is_active = i < size
        coord_logits = jnp.where(is_active, jnp.log(jnp.sum(vecs ** 2, -1)), 0)
        idx = jax.random.categorical(key0, coord_logits)
        new_vecs = _orthogonal(vecs, jnp.where(is_active, idx, 0))
        cond = (jnp.arange(n) < (size - i - 1)) & ~samples
        new_vecs = jnp.where(cond, new_vecs, 0)
        vecs = jnp.where(is_active, new_vecs, vecs)
        sample = samples | ((jnp.arange(d) == idx) & is_active)
        return i + 1, vecs, sample, key

    _, _, sample, _ = jax.lax.while_loop(lambda val: val[0] < size, _body,
                                         (0, eigen_vecs, jnp.zeros(n, dtype=jnp.bool_), key))
    return sample


@jax.jit
def dpp(l: Array,
        key: jax.random.PRNGKeyArray
        ) -> Array:
    """ Determinantal Point Process with for a symmetric matrix. This function returns a bool vector corresponding to
    the output set.
    """

    key, key0 = jax.random.split(key)
    val, vec = jnp.linalg.eigh(l)
    edpp_indices = jax.random.bernoulli(key0, val / (val + 1))
    return _sample_edpp(vec, edpp_indices, key)


def _elementary_symmetric_polynomials(eigen_vals: Array,
                                      k: int
                                      ) -> Array:
    n = eigen_vals.shape[0]
    e = jnp.zeros((n + 1, k + 1))
    e = e.at[:, 0].set(1.0)

    def f(carry: Array, x: Array):
        return carry.at[x, 1:].set(carry[x - 1, 1:] + eigen_vals[x - 1] * carry[x - 1, :-1]), None

    e, _ = jax.lax.scan(f, e, jnp.arange(1, n + 1))
    return e


def _kdpp_indices(eigen_vals: Array,
                  k: int,
                  key: jax.random.PRNGKeyArray
                  ) -> Array:
    n = eigen_vals.shape[0]
    e = _elementary_symmetric_polynomials(eigen_vals, k)

    def _body(val):
        l, n, sample, key = val
        key, key0 = jax.random.split(key)
        i = jax.random.bernoulli(key0, eigen_vals[n] * e[n - 1, l - 1] / e[n, l]).astype(int)
        l = l - i
        sample = sample.at[n - 1].set(i)
        return l, n - 1, sample, key

    sample = jnp.zeros(n, dtype=int)
    _, _, sample, _ = jax.lax.while_loop(lambda val: val[0] > 0, _body, (k, n - 1, sample, key))
    return sample


@functools.partial(jax.jit, static_argnums=1)
def kdpp(l: Array,
         k: int,
         key: jax.random.PRNGKeyArray
         ) -> Array:
    """ k-DPP with for a symmetric matrix. This function returns a bool vector corresponding to the output set.
    """

    key, key0 = jax.random.split(key)
    val, vec = jnp.linalg.eigh(l)
    indices = _kdpp_indices(val, k, key0)
    return _sample_edpp(vec, indices, key)
