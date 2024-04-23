directions = [
    [
        0x80000000, 0x40000000, 0x20000000, 0x10000000,
        0x08000000, 0x04000000, 0x02000000, 0x01000000,
        0x00800000, 0x00400000, 0x00200000, 0x00100000,
        0x00080000, 0x00040000, 0x00020000, 0x00010000,
        0x00008000, 0x00004000, 0x00002000, 0x00001000,
        0x00000800, 0x00000400, 0x00000200, 0x00000100,
        0x00000080, 0x00000040, 0x00000020, 0x00000010,
        0x00000008, 0x00000004, 0x00000002, 0x00000001,
    ],
    [
        0x80000000, 0xc0000000, 0xa0000000, 0xf0000000,
        0x88000000, 0xcc000000, 0xaa000000, 0xff000000,
        0x80800000, 0xc0c00000, 0xa0a00000, 0xf0f00000,
        0x88880000, 0xcccc0000, 0xaaaa0000, 0xffff0000,
        0x80008000, 0xc000c000, 0xa000a000, 0xf000f000,
        0x88008800, 0xcc00cc00, 0xaa00aa00, 0xff00ff00,
        0x80808080, 0xc0c0c0c0, 0xa0a0a0a0, 0xf0f0f0f0,
        0x88888888, 0xcccccccc, 0xaaaaaaaa, 0xffffffff,
    ],
    [
        0x80000000, 0xc0000000, 0x60000000, 0x90000000,
        0xe8000000, 0x5c000000, 0x8e000000, 0xc5000000,
        0x68800000, 0x9cc00000, 0xee600000, 0x55900000,
        0x80680000, 0xc09c0000, 0x60ee0000, 0x90550000,
        0xe8808000, 0x5cc0c000, 0x8e606000, 0xc5909000,
        0x6868e800, 0x9c9c5c00, 0xeeee8e00, 0x5555c500,
        0x8000e880, 0xc0005cc0, 0x60008e60, 0x9000c590,
        0xe8006868, 0x5c009c9c, 0x8e00eeee, 0xc5005555,
    ],
    [
        0x80000000, 0xc0000000, 0x20000000, 0x50000000,
        0xf8000000, 0x74000000, 0xa2000000, 0x93000000,
        0xd8800000, 0x25400000, 0x59e00000, 0xe6d00000,
        0x78080000, 0xb40c0000, 0x82020000, 0xc3050000,
        0x208f8000, 0x51474000, 0xfbea2000, 0x75d93000,
        0xa0858800, 0x914e5400, 0xdbe79e00, 0x25db6d00,
        0x58800080, 0xe54000c0, 0x79e00020, 0xb6d00050,
        0x800800f8, 0xc00c0074, 0x200200a2, 0x50050093,
    ],
    [
        0x80000000, 0x40000000, 0x20000000, 0xb0000000,
        0xf8000000, 0xdc000000, 0x7a000000, 0x9d000000,
        0x5a800000, 0x2fc00000, 0xa1600000, 0xf0b00000,
        0xda880000, 0x6fc40000, 0x81620000, 0x40bb0000,
        0x22878000, 0xb3c9c000, 0xfb65a000, 0xddb2d000,
        0x78022800, 0x9c0b3c00, 0x5a0fb600, 0x2d0ddb00,
        0xa2878080, 0xf3c9c040, 0xdb65a020, 0x6db2d0b0,
        0x800228f8, 0x400b3cdc, 0x200fb67a, 0xb00ddb9d,
    ]
]

MASK_32BIT = 0xffffffff
MASK_64BIT = 0xffffffffffffffff
SOBOL_SCALE_32BIT = 1.0/(1<<32)

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

def nested_uniform_scramble_base2(x, seed):
    x = reverse_bits(x)
    x = vegdahl_permutation(x, seed) # laine_karras_permutation(x, seed);
    x = reverse_bits(x)
    return x

def sobol(index: int, dim: int) -> int:
    assert dim <= 4
    X = 0
    for bit in range(32):
        mask = (index >> bit) & 1
        X ^= mask * directions[dim][bit]
    return X

def sobol_float(index: int, dim: int, seed: int = 0) -> float:
    _ = seed
    return sobol(index, dim) * SOBOL_SCALE_32BIT

def sobol_float_rds(i, dim, seed=0) -> float:
    seed = hash(seed)
    scramble = hash_combine(seed, hash(dim))
    return (sobol(i,dim) ^ scramble) * SOBOL_SCALE_32BIT

def sobol_float_owen(i, dim, seed=0) -> float:
    seed = hash(seed)
    index = nested_uniform_scramble_base2(i, seed)
    return nested_uniform_scramble_base2(sobol(index, dim), hash_combine(seed, dim)) * SOBOL_SCALE_32BIT