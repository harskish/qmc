from pathlib import Path
from .direction_numbers import get_direction_numbers

directions = None
def get_directions():
    global directions
    if directions is None:
        directions = get_direction_numbers(dims=100)

MASK_32BIT = 0xffffffff
MASK_64BIT = 0xffffffffffffffff
ONE_OVER_U32_MAX = 1.0/(1<<32)

# Explicitly truncate to 64bits
# Python has no unsigned integer datatype
def i64(v: int):
    return v & MASK_64BIT

def i32(v: int):
    return v & MASK_32BIT

# Somewhat similar to boost::hash_combine
def hash_combine(seed: int, v: int) -> int:
    return i32(seed ^ (v + (seed << 6) + (seed >> 2)))

def hash(x: int):
    # finalizer from murmurhash3
    x = i32(x)
    x = x ^ (x >> 16)
    x = i32(x * 0x85ebca6b)
    x = x ^ (x >> 13)
    x = i32(x * 0xc2b2ae35)
    x = x ^ (x >> 16)
    return x

# Complexity is loglogX?
def reverse_bits(x: int) -> int:
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1))
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2))
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4))
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8))
    return i32((x >> 16) | (x << 16))

# Improved LK variant by Nathan Vegdahl:
# psychopath.io/post/2021_01_30_building_a_better_lk_hash
def vegdahl_permutation(x, seed):
    x ^= x * 0x3d20adea
    x += seed
    x *= (seed >> 16) | 1
    x ^= x * 0x05526c56
    x ^= x * 0x53a22864
    return i32(x)

# Source: Burley2020, Listing 2
# Example usage: https://www.shadertoy.com/view/wlyyDm
def nested_uniform_scramble_base2(x, seed):
    x = reverse_bits(x)
    x = vegdahl_permutation(x, seed) # laine_karras_permutation(x, seed);
    x = reverse_bits(x)
    return x

# "Ground truth" Owen scrambling
def owen_scramble_reference(n: int, seed: int) -> int:
    in_bits = n
    out_bits = n
    for bit in range(32):
        high_mask = ~(1 << bit)
        hashed = hash_combine(hash(in_bits & high_mask), seed) # seed already hashed
        out_bits ^= hashed & (1 << bit)
    return out_bits

def sobol_int(index: int, dim: int) -> int:
    get_directions()
    assert dim < len(directions)
    X = 0
    for bit in range(32): # log2(N)
        mask = (index >> bit) & 1
        X ^= mask * directions[dim][bit]
    return X

def sobol(index, dim, seed=0, N=None) -> float:
    _ = seed
    _ = N
    return sobol_int(index, dim) * ONE_OVER_U32_MAX

def sobol_rds(i, dim, seed=0, N=None) -> float:
    _ = N
    seed = hash(seed)
    scramble = hash_combine(seed, hash(dim))
    return (sobol_int(i,dim) ^ scramble) * ONE_OVER_U32_MAX

def sobol_owen_vegdahl(i, dim, seed=0, N=None) -> float:
    _ = N
    seed = hash(seed)
    index = nested_uniform_scramble_base2(i, seed)
    return nested_uniform_scramble_base2(sobol_int(index, dim), hash_combine(seed, dim)) * ONE_OVER_U32_MAX

def sobol_owen_ref(i, dim, seed=0, N=None) -> float:
    _ = N
    seed = hash(seed)
    index = owen_scramble_reference(i, seed)
    return owen_scramble_reference(sobol_int(index, dim), hash_combine(seed, dim)) * ONE_OVER_U32_MAX