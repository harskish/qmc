from typing import Callable
import numpy as np
from math import sqrt
from functools import lru_cache
import matplotlib.pyplot as plt
import struct

# [1]: pbr-book.org/3ed-2018/Sampling_and_Reconstruction/The_Halton_Sampler
# [2]: psychopath.io/post/2020_04_14_building_a_sobol_sampler

BIG_PRIME = 15487313

def seeded_randint(seed):
    return np.random.RandomState(seed=seed).randint(0, 2<<63, dtype=np.uint64).item()

def seeded_rand(seed):
    return np.random.RandomState(seed=seed).rand()

# TODO: 64-bit version?
# Somewhat similar to boost::hash_combine
def hash_combine(hash: int, v: int) -> int:
    return hash ^ (v + (hash << 6) + (hash >> 2))

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

def halton_gen(b_idx):
    b = nth_prime(b_idx)
    d = 1
    n = 0
    while True:
        x = d - n
        if x == 1:
            n = 1
            d *= b
        else:
            y = d // b
            while x <= y:
                y //= b
            n = (b + 1) * y - x
        yield n / d

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
def cranley_patterson_rotation():
    pass

# Reinterpret float bits as int
def float_as_int(a: float):
    return struct.unpack('@Q', struct.pack('@d', a))[0]

# Reinterpret int bits as float
def int_as_float(a: int):
    return struct.unpack('@d', struct.pack('@Q', a))[0]

# This only works for Sobol/(0,2) samples before 2^-64 division
# (when the values are base2 integers)
# pbr-book.org/3ed-2018/Sampling_and_Reconstruction/(0,_2)-Sequence_Sampler
def random_digit_scrambling(a: float, mask: int):
    #mask_repr = format(mask, '#066b')
    #a_repr = format(float_as_int(a), '#066b')
    res = int_as_float(float_as_int(a) ^ mask)
    #res_repr = format(float_as_int(res), '#066b')
    return res

def owen_scrambling():
    pass

def halton(b_ind: int, i: int, N: int = None):
    _ = N # ignored
    b = nth_prime(b_ind)
    return radical_inverse(b, i+1)

def hammersley(b_ind: int, i: int, N: int):
    if b_ind == 0:
        return i / N
    return halton(b_ind - 1, i)

def uber_sampler(
    func: Callable, # qmc sampler funcion
    b_ind: int,
    i: int,
    rds_seed: int = None, # random digit scrambling, only for Sobol
    owen_seed: int = None, # Owen scrambling
    cp_seed: int = None, # Cranley-Patterson rotation
    N: int = None,
):
    v = func(b_ind, i, N)
    if rds_seed is not None:
        print('Not implemented!')
    if owen_seed is not None:
        print('Not implemented!')
    if cp_seed is not None:
        seed = hash_combine(b_ind, cp_seed)
        #seed = b_ind * BIG_PRIME + cp_seed
        v = (v + seeded_rand(seed)) % 1
    return v

def plot_2d(func, b1, b2, *args, N=128, **kwargs):
    xs = [uber_sampler(func, b1, i, *args, **kwargs, N=N) for i in range(N)]
    ys = [uber_sampler(func, b2, i, *args, **kwargs, N=N) for i in range(N)]
    plt.figure(figsize=(4, 4))
    plt.plot(xs, ys, 'bo')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(
        f'{func.__name__} ({nth_prime(b1)},{nth_prime(b2)})\n' + 
        f"cp{kwargs.get('cp_seed', '-none')}, rds{kwargs.get('rds_seed', '-none')}, owen{kwargs.get('owen_seed', '-none')}"
    )

def plot_2d_broken(func, b1, b2, *args, N=128, **kwargs):
    # Intentionally misuse generator (consecutive bases, sample indiceds {0,1})
    xs = [uber_sampler(func, i, 0, *args, **kwargs, N=N) for i in range(N)]
    ys = [uber_sampler(func, i, 1, *args, **kwargs, N=N) for i in range(N)]
    plt.figure(figsize=(4, 4))
    plt.plot(xs, ys, 'bo')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(
        f'Broken-{func.__name__} ({nth_prime(b1)},{nth_prime(b2)})\n' + 
        f"cp{kwargs.get('cp_seed', '-none')}, rds{kwargs.get('rds_seed', '-none')}, owen{kwargs.get('owen_seed', '-none')}"
    )

if __name__ == '__main__':
    plot_2d(halton, 0, 1)
    plot_2d_broken(halton, 0, 1)
    plt.show()
    print('Done')