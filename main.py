from collections import defaultdict
from typing import Callable
import numpy as np
from math import sqrt
from functools import lru_cache
import matplotlib.pyplot as plt
import struct

# [1]: pbr-book.org/3ed-2018/Sampling_and_Reconstruction/The_Halton_Sampler
# [2]: psychopath.io/post/2020_04_14_building_a_sobol_sampler
# [3]: github.com/openturns/openturns/issues/1601#issuecomment-702728408
# [4]: github.com/scipy/scipy/pull/10844
# [5]: jcgt.org/published/0009/04/01/

BIG_PRIME = 15487313
MASK_32BIT = 0xffffffff; assert MASK_32BIT.bit_length() == 32
MASK_64BIT = 0xffffffffffffffff; assert MASK_64BIT.bit_length() == 64

import burley2020

# Explicitly truncate to 64bits
# Python has no unsigned integer datatype
def i64(v: int):
    return v & MASK_64BIT

def i32(v: int):
    return v & MASK_32BIT

def seeded_randint(seed):
    return np.random.RandomState(seed=seed).randint(0, 1<<63, dtype=np.uint64).item()

def seeded_rand(seed):
    return np.random.RandomState(seed=seed).rand()

# TODO: u64 version?
def hash(x: int):
    # finalizer from murmurhash3
    x = i32(x ^ (x >> 16))
    x = i32(x * (0x85ebca6b))
    x = i32(x ^ (x >> 13))
    x = i32(x * (0xc2b2ae35))
    x = i32(x ^ (x >> 16))
    return x

# TODO: 64-bit version?
# Somewhat similar to boost::hash_combine
def hash_combine(hash: int, v: int) -> int:
    return i32(hash ^ (v + (hash << 6) + (hash >> 2)))

def hash_combine_naive(hash: int, v: int) -> int:
    return i32(hash + BIG_PRIME * v)

def is_prime(a):
    if a < 2:
        return False
    for x in range(2, int(sqrt(a)) + 1):
        if a % x == 0:
            return False
    return True

@lru_cache
def nth_prime(n):
    """ Returns the Nth prime (zero-indexed) """
    if n == 0:
        return 2
    if n == 1:
        return 3
    i = nth_prime(n - 1) # previous
    while True:
        i += 2 # skip even
        if is_prime(i):
            return i

def get_primes(N):
    return [nth_prime(i) for i in range(N)]

def radical_inverse(b: int, i: int):
    f = 1
    r = 0
    while i > 0: # log_b(i) steps?
        f = f / b
        r += f * (i % b)
        i = i // b
    return r

# Used to create decorrelated sample sequences.
# Increases variance (see [2]).
# Computes {p_i + s | mod 1} for all i.
def cranley_patterson_rotation(v, dim, seed):
    return (v + seeded_rand(hash_combine(seed, hash(dim)))) % 1

# Reinterpret float bits as int
def float_as_int(a: float):
    return struct.unpack('@Q', struct.pack('@d', a))[0]

# Reinterpret int bits as float
def int_as_float(a: int):
    return struct.unpack('@d', struct.pack('@Q', a))[0]

def halton(dim: int, i: int, N: int = None, seed: int = None):
    _ = N # ignored
    if seed:
        raise NotImplementedError()
    b = nth_prime(dim)
    return radical_inverse(b, i) # include zero [3,4]

def hammersley(dim: int, i: int, N: int, seed: int = None):
    if seed:
        raise NotImplementedError()
    if dim == 0:
        return i / N
    return halton(dim - 1, i)

# Getting favorable convergence with high-dimensional Sobol sequences requires huge sample counts (Burley2020 sec. 2.3)
# => instead use low-dimensional 4D sequence, pad with shuffled RQMC point sets (here: Owen-shuffled)
_sobol_cache = defaultdict(list)
def sobol_impl(variant: str, dim: int, i: int, N: int, seed: int = None):
    global _sobol_cache
    seed = seed or 0
    key = f'{variant}_{dim}_{seed}'
    if len(_sobol_cache[key]) < i+1:
        n_samp = max(N, 2*len(_sobol_cache[key]))
        _sobol_cache[key] = burley2020.sample(variant, n=n_samp, dim=dim, seed=seed) # scramble seed?
    return _sobol_cache[key][i]

def sobol_owen(*args, **kwargs):
    return sobol_impl('sobol_owen', *args, **kwargs)

def sobol_rds(*args, **kwargs):
    return sobol_impl('sobol_rds', *args, **kwargs)

# No scrambling or permutation?
def sobol(*args, **kwargs):
    return sobol_impl('sobol', *args, **kwargs)

def uber_sampler(
    func: Callable, # qmc sampler funcion
    dim: int,
    i: int,
    seed: int = 0, # scrambling/permutation seed (rds, Owen)
    cp_seed: int = None, # Cranley-Patterson rotation
    N: int = None,
):
    v = func(dim, i, N, seed)
    if cp_seed is not None:
        v = cranley_patterson_rotation(v, dim, cp_seed)
    return v

def plot_2d(func, b1, b2, *args, N=256, **kwargs):
    xs = [uber_sampler(func, b1, i, *args, **kwargs, N=N) for i in range(N)]
    ys = [uber_sampler(func, b2, i, *args, **kwargs, N=N) for i in range(N)]
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, 'bo')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(
        f'{func.__name__} dims=({b1},{b2})\n' + 
        f"cp{kwargs.get('cp_seed', '-none')}, rds{kwargs.get('rds_seed', '-none')}, owen{kwargs.get('owen_seed', '-none')}"
    )

def plot_2d_broken(func, b1, b2, *args, N=256, **kwargs):
    # Intentionally misuse generator (consecutive bases, sample indiceds {b1,b2})
    xs = [uber_sampler(func, i, b1, *args, **kwargs, N=N) for i in range(N)]
    ys = [uber_sampler(func, i, b2, *args, **kwargs, N=N) for i in range(N)]
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, 'bo')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(
        f'Broken-{func.__name__} dims=({b1},{b2})\n' + 
        f"cp{kwargs.get('cp_seed', '-none')}, seed{kwargs.get('seed', '-none')}"
    )

if __name__ == '__main__':
    plot_2d(sobol_owen, 0, 1, seed=0)
    plot_2d(sobol_owen, 0, 2, seed=0)
    plot_2d(sobol_owen, 1, 2, seed=0)
    plot_2d(sobol_owen, 0, 3, seed=0)
    plot_2d(sobol_owen, 1, 3, seed=0)
    plot_2d(sobol_owen, 2, 3, seed=0)
    #plot_2d(sobol_owen, 0, 1, seed=0, cp_seed=0)
    #plot_2d(sobol_owen, 0, 1, seed=1)
    #plot_2d(sobol_rds, 0, 1, seed=0)
    #plot_2d(sobol_rds, 0, 1, seed=1)
    #plot_2d(sobol, 0, 1)
    #plot_2d(halton, 0, 1)
    #plot_2d(hammersley, 0, 1)
    ##plot_2d(hammersley, 0, 1, cp_seed=0)
    #plot_2d_broken(halton, 1, 2)
    plt.show()
    print('Done')

# Want to optimize for:
# - low multidimensional discrepancy
# - high minimum sample distance
# - efficient computation (preferably in base-2)
# - random access

# Sobol:
# - Implements 1D-stratification-preserving permutations of the binary van der Corput sequence
#   => achieved through nested binary permutations
#   => LK-hashing also preserves 1D stratification (for all sample counts)
# - Additionally, the permutaitons are chosen to minimize multidimensional discrepancy
# - Permutations are stored as a matrix of 'direction numbers'

# Shuffling (randomize index): creates uncorrelated variants of given low-discrepancy sequence
# - Decorrelates the ordering of samples for padding without changing other poperties
# - used to stich several sequences together along the dimension axis
#   => avoids high-dim sequences, which have large sample count requirements for favorable convergence [5, sec. 2.3]
# - implements padding, which can also be done with pure random noise
# - Can be implemented using Owen shuffling (!= Owen scrambling)

# Scrambling (randomize value): introduces randomness into QMC sequence
# - Improves properties of the sequence (breaks up structured patterns, improves convergence)
# - aka. randomization
# - rds commonly used, easy to implement
# - Owen-scrambling (aka. Nested Uniform Scrambling) better, can actually improve convergence rate!
# - applies permutations to the elementary intervals of the output domain
# - Just scrambling not enough when padding (sequences will still be quite correlated)

# Global samplers:
# - use a single sequence over the image plane in rendering
# - does not mean that pixels have to be rendered in a LDS order!
#   => if the sequence supports skipping, then pixels can still be processed in an arbitrary order
# - might leave artifacts in the image plane

# Sample decorrelation:
# - Shufflig (e.g. Owen): works well
# - Random sample offsetting: might be slow, depending on sampler
# - Cranley-patterson rotation: increases variance