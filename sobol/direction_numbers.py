import numpy as np
from pathlib import Path
from tqdm import trange

root = Path(__file__).parent

def get_burley_direction_numbers():
    # 5 dims, 32bit
    return [
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

# github.com/scipy/scipy/blob/8431e12346ac9564e244bfce920dc8a04bcabbd9/scipy/stats/_sobol.pyx#L82
def convert_direction_numbers(outfile):
    import pandas as pd

    # read in file content
    with open(root / "new-joe-kuo-6.21201", "r") as f:
        lines = f.readlines()

    rows = []

    # parse data from file line by line
    for l in lines[1:]:
        nums = [int(n) for n in l.replace(" \n", "").split()]
        d, s, a = nums[:3]
        vs = {f"v{i}": int(v) for i,v in enumerate(nums[3:])}
        rows.append({"d": d, "s": s, "a": a, **vs})


    # read in as dataframe, explicitly use zero values
    df = pd.DataFrame(rows).fillna(0).astype(int)

    # perform conversion
    df["poly"] = 2 * df["a"] + 2 ** df["s"] + 1

    # ensure columns are properly ordered
    vs = df[[f"v{i}" for i in range(18)]].values

    # add the degenerate d=1 column (not included in the data file)
    vs = np.vstack([vs[0][np.newaxis, :], vs])
    poly = np.concatenate([[1], df["poly"].values])

    # save as compressed .npz file to minimize size of distribution
    np.savez_compressed(outfile, vinit=vs, poly=poly)

# Build full matrix of shape [dims][bits]
# github.com/scipy/scipy/blob/8431e12346ac9564e244bfce920dc8a04bcabbd9/scipy/stats/_sobol.pyx#L241
def build_matrix(poly, vinit, bits=32, dims=1111):
    assert 0 < dims <= vinit.shape[0]
    v = np.zeros((dims, bits), dtype={32: np.uint32, 64: np.uint64}[bits])
    if dims > 1111:
        print('Warning: property A only holds for first 1111 dimensions')

    # first row of v is all 1s
    v[0, :] = 1

    # Remaining rows of v (row 2 through dim, indexed by [1:dim])
    for d in trange(1, dims):
        p = poly[d]
        m = p.item().bit_length() - 1

        # First m elements of row d comes from vinit
        v[d, :m] = vinit[d, :m]

        # Fill in remaining elements of v as in Section 2 (top of pg. 90) of:
        #
        # P. Bratley and B. L. Fox. Algorithm 659: Implementing sobol's
        # quasirandom sequence generator. ACM Trans.
        # Math. Softw., 14(1):88-100, Mar. 1988.
        #
        for j in range(m, bits):
            newv = v[d, j - m]
            pow2 = 1
            for k in range(m):
                pow2 = pow2 << 1
                if (p >> (m - 1 - k)) & 1:
                    newv = newv ^ (pow2 * v[d, j - k - 1])
            v[d, j] = newv

    # Multiply each column of v by power of 2:
    # v * [2^(bits-1), 2^(bits-2),..., 2, 1]
    pow2 = 1
    for d in range(bits):
        for i in range(dims):
            v[i, bits - 1 - d] *= pow2
        pow2 = pow2 << 1
    
    return v.tolist()

def get_direction_numbers(dims=1111):
    outfile = root / '_sobol_direction_numbers.npz'
    if not outfile.is_file():
        convert_direction_numbers(outfile)
    data = np.load(outfile)
    return build_matrix(data['poly'], data['vinit'], dims=dims) #21201