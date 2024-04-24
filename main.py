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

BIG_PRIME = 15487313

from sobol import i32, hash, hash_combine, sobol, sobol_owen, sobol_rds

def seeded_randint(seed):
    return np.random.RandomState(seed=seed).randint(0, 1<<63, dtype=np.uint64).item()

def seeded_rand(seed):
    return np.random.RandomState(seed=seed).rand()

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

def halton(i: int, dim: int, seed: int = 0):
    if seed:
        raise NotImplementedError()
    b = nth_prime(dim)
    return radical_inverse(b, i) # include zero [3,4]

def hammersley(i: int, dim: int, N: int, seed: int = 0):
    if seed:
        raise NotImplementedError()
    if dim == 0:
        return i / N
    return halton(i, dim - 1)

# Getting favorable convergence with high-dimensional Sobol sequences requires huge sample counts (Burley2020 sec. 2.3)
# => instead use low-dimensional 4D sequence, pad with shuffled RQMC point sets (here: Owen-shuffled)
_sobol_cache = defaultdict(list)
def sobol_cpp_impl(variant: str, i: int, dim: int, seed: int = 0):
    global _sobol_cache
    key = f'{variant}_{dim}_{seed}'
    if len(_sobol_cache[key]) < i+1:
        import burley2020
        n_samp = max(i+1, 2*len(_sobol_cache[key]))
        _sobol_cache[key] = burley2020.sample(variant, n=n_samp, dim=dim, seed=seed) # scramble seed?
    return _sobol_cache[key][i]

def sobol_owen_cpp(*args, **kwargs):
    return sobol_cpp_impl('sobol_owen', *args, **kwargs)

def sobol_rds_cpp(*args, **kwargs):
    return sobol_cpp_impl('sobol_rds', *args, **kwargs)

def sobol_cp(i: int, dim: int, seed: int):
    return cranley_patterson_rotation(sobol(i, dim, seed), dim, seed)

def plot_2d(func, dim1, dim2, *args, N=1024, **kwargs):
    if func.__name__ == 'hammersley':
        kwargs['N'] = N
    xs = [func(i, dim1, *args, **kwargs) for i in range(N)]
    ys = [func(i, dim2, *args, **kwargs) for i in range(N)]
    plt.figure(figsize=(8, 8))
    plt.plot(xs, ys, 'bo')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f'{func.__name__} dims=({dim1},{dim2})\nseed{kwargs.get("seed", 0)}')

if __name__ == '__main__':
    pass
    #plot_2d(sobol, 0, 1, seed=0)
    #plot_2d(sobol, 1, 2, seed=0)
    #plot_2d(sobol, 0, 4, seed=0)
    #plot_2d(sobol_rds, 0, 1, seed=0)
    #plot_2d(sobol_rds, 0, 1, seed=123)
    #plot_2d(sobol_rds, 0, 1, seed=999)
    #plot_2d(sobol_owen, 0, 1, seed=0)
    #plot_2d(sobol_owen, 0, 1, seed=123)
    #plot_2d(sobol_owen, 0, 1, seed=999)
    #plot_2d(halton, 0, 1, seed=0)
    #for i in range(0, 20):
    #    plot_2d(hammersley, 0, i+1, seed=0)
    #for i in range(0, 20):
    #    plot_2d(hammersley, i, i+1, seed=0)
    plot_2d(sobol_owen, 14, 15, seed=0)
    plot_2d(sobol_owen, 14, 15, seed=1)
    plot_2d(sobol_owen, 14, 15, seed=2)
    plot_2d(sobol_owen, 14, 15, seed=3)
    plot_2d(sobol_owen, 14, 15, seed=4)
    
    for i in range(0, 15):
        plot_2d(sobol, i, i+1, seed=0)
    for i in range(0, 15):
        plot_2d(sobol_owen, i, i+1, seed=0)
    print('Done')