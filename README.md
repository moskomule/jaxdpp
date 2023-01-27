# jaxdpp

JAX implementation of determinantal point processes

## requirements

```
jax
jaxtyping
```

## usage

Currently, an input matrix is expected to be symmetric (thus `eigh` is used internally).

```python
import jax
from sampling import dpp, kdpp

l = your_symmetric_matrix
dpp(l, jax.random.PRNGKey(0))
# [True, False, True, True, ...]

kdpp(l, jax.random.PRNGKey(0), k=3)
# [True, True, True, False, False, ...]
```

## acknowledgement

This implementation is based on [Kulesza and Taskar, 2012]() and [tensorflow-probability]().
