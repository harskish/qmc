import time
from collections import defaultdict
import numpy as np
from math import sqrt
from functools import lru_cache
import pyviewer
from pyviewer.toolbar_viewer import AutoUIViewer
from pyviewer.params import *
from imgui_bundle import implot, implot3d
import glfw
from sobol import *

assert pyviewer.__version__ >= '2.0.0', 'pyviewer 2.0.0+ required'

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
# - Shuffling (e.g. Owen): works well
# - Random sample offsetting: might be slow, depending on sampler
# - Cranley-patterson rotation: increases variance

# https://github.com/sparks-baird/self-driving-lab-demo/blob/main/notebooks/escience/1.0-traditional-doe-vs-bayesian.ipynb
print('TODO: measure discrepancy with scipy.stats.qmc.discrepancy')

#BIG_PRIME = 15487313 # = nth_prime(1_000_091)
BIG_PRIME =  7778777 # = nth_prime(525_831)

from sobol import *

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

@lru_cache(maxsize=None)
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
    exp = 1
    rev = 0
    while i > 0:
        exp = exp / b # left of decimal => negative powers
        rev += exp * (i % b) # lsd
        i = i // b # next power of b
    return rev

# Generalized golden ratio
# d=1: golden ratio, d=2: plastic ratio
@lru_cache
def phi(d, iter=20):
  x = 2.0
  for _ in range(iter):
    x = pow(1+x, 1/(d+1))
  return x

@lru_cache
def cranley_patterson_offset(dim, seed):
    return seeded_rand(hash_combine(seed, hash(dim)))

# Used to create decorrelated sample sequences.
# Increases variance (see [2]).
# Computes {p_i + s | mod 1} for all i.
def cranley_patterson_rotation(v, dim, seed):
    return (v + cranley_patterson_offset(dim, seed)) % 1

def halton(i: int, dim: int, seed: int = 0, N: int = None):
    _ = N
    b = nth_prime(dim)
    return radical_inverse(b, i) # include zero [3,4]

def leaped_halton(i: int, dim: int, seed: int = 0, N: int = None):
    _ = N
    b = nth_prime(dim)
    leap = nth_prime(seed + 79) # start at 409
    return radical_inverse(b, i*leap) # include zero [3,4]

def hammersley(i: int, dim: int, N: int, seed: int = 0):
    if dim == 0:
        return i / N
    return halton(i, dim - 1)

# [1] extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
# [2] martysmods.com/a-better-r2-sequence/
def Weyl(i: int, dim: int, N: int, seed: int = 0, max_dim: int = 2):
    p = pow(1/phi(max_dim), dim+1) % 1 # inverse power of plastic number
    return cranley_patterson_rotation((1-p)*(i+1), dim, seed) # p => (1-p), see [2]

def R2(i: int, dim: int, N: int, seed: int = 0):
    return Weyl(i, dim, N, seed, max_dim=2)

def R3(i: int, dim: int, N: int, seed: int = 0):
    return Weyl(i, dim, N, seed, max_dim=3)

# Getting favorable convergence with high-dimensional Sobol sequences requires huge sample counts (Burley2020 sec. 2.3)
# => instead use low-dimensional 4D sequence, pad with shuffled RQMC point sets (here: Owen-shuffled)
_sobol_cache = defaultdict(list)
def sobol_cpp_impl(variant: str, i: int, dim: int, seed: int = 0, N: int = None):
    global _sobol_cache
    key = f'{variant}_{dim}_{seed}'
    if len(_sobol_cache[key]) < i+1:
        import burley2020
        n_samp = max(i+1, 2*len(_sobol_cache[key]))
        _sobol_cache[key] = burley2020.sample(variant, n=n_samp, dim=dim, seed=seed) # scramble seed?
    return _sobol_cache[key][i]

def sobol_owen_Burley(*args, **kwargs):
    return sobol_cpp_impl('sobol_owen', *args, **kwargs)

def sobol_rds_Burley(*args, **kwargs):
    return sobol_cpp_impl('sobol_rds', *args, **kwargs)

def faure05_Burley(*args, **kwargs):
    return sobol_cpp_impl('faure05', *args, **kwargs)

def faure05_owen_Burley(*args, **kwargs):
    return sobol_cpp_impl('faure05_owen', *args, **kwargs)

def sobol_cp(i: int, dim: int, seed: int, N: int):
    _ = N
    return cranley_patterson_rotation(sobol(i, dim, seed), dim, seed)

def cascaded_sobol_online(i: int, dim: int, seed: int, N: int):
    import paulin2021
    return paulin2021.sample(i, dim, seed)

def cascaded_sobol(i: int, dim: int, seed: int, N: int):
    import paulin2021
    return paulin2021.sample_ref(i, dim, seed)

def random(i: int, dim: int, seed: int, N: int):
    return seeded_rand(hash_combine(i, hash_combine(dim, seed)))

def murmur(i: int, dim: int, seed: int, N: int):
    return hash(hash_combine(i, hash_combine(dim, seed))) * ONE_OVER_U32_MAX

@strict_dataclass
class State(ParamContainer):
    N: Param = IntParam('Samples', 128, 1, 2048)
    seq: Param = EnumParam('Sequence', sobol_owen_vegdahl, [
        murmur,              # Murmurhash3 finalizer (non-LDS)
        sobol,               # Standard Sobol (no scrambling or shuffling)
        sobol_cp,            # Sobol + Cranley Patterson rotation (scrambling of values)
        sobol_rds,           # Sobol + random digit scrambling (of values)
        sobol_owen_ref,      # 'Ground-truth' Owen scrambling (value scrambling + idx shuffling) [buggy?!]
        sobol_owen_vegdahl,  # Fast Owen scrambling using Vegdahl hash
        sobol_owen_Burley,   # Burley2020 reference C++ Owen-scrambled Sobol
        cascaded_sobol,      # Paulin2021 C++ impl
        faure05_Burley,      # Burley2020 reference C++ Faure05 sampler
        faure05_owen_Burley, # Burley2020 reference C++ Faure05 + Owen scrambling
        hammersley,          # Standard Hammersley (no scrambling or shuffling)
        halton,              # Standard Halton (no scrambling or shuffling)
        leaped_halton,       # Leaped Halton (seeded prime-sized strides on idx)
        R2,                  # R2 sequence (2D "Roberts Sequence", i.e. Golden ratio Weyl sequence)
        R3,                  # R3 sequence (3D "Roberts Sequence", i.e. Golden ratio Weyl sequence)
    ], lambda f: f.__name__)
    seed: Param = IntParam('Seed', 0, 0, 99, buttons=True)
    dim1: Param = IntParam('X dimension', 0, 0, 50, buttons=True)
    dim2: Param = IntParam('Y dimension', 1, 0, 50, buttons=True)
    dim3: Param = IntParam('Z dimension', 2, 0, 50, buttons=True)
    plot3d: Param = BoolParam('3D', False)
    
class Viewer(AutoUIViewer):
    def setup_state(self):
        implot3d.create_context()
        self.draw_scale_buttons = False
        self.arr_create_time = 0
        self.state = State()

    @lru_cache(maxsize=1)
    def get_data(self, fun, N, seed, dim1, dim2, dim3):
        return np.array([fun(i, dim1, seed=seed, N=N) for i in range(N)]), \
               np.array([fun(i, dim2, seed=seed, N=N) for i in range(N)]), \
               np.array([fun(i, dim3, seed=seed, N=N) for i in range(N)])
    
    def draw_pre(self):
        state = self.state
        W, H = glfw.get_window_size(self.v._window)
        style = imgui.get_style()
        avail_h = H - self.menu_bar_height - 2*style.window_padding.y - self.pad_bottom
        avail_w = W - self.toolbar_width
        t0 = time.monotonic()
        xs, ys, zs = self.get_data(state.seq, state.N, state.seed, state.dim1, state.dim2, state.dim3) # cached
        self.arr_create_time = time.monotonic() - t0
        
        if state.plot3d:
            if implot3d.begin_plot('LDS##3D', size=(avail_w, avail_h)):
                implot3d.set_next_marker_style(size=6*self.ui_scale)
                implot3d.setup_box_scale(1.2, 1.2, 1.2)
                implot3d.plot_scatter('Sequence', xs, ys, zs)
                implot3d.end_plot()
        else:
            if implot.begin_plot('LDS##2D', size=(avail_w, avail_h), flags=implot.Flags_.equal):
                implot.set_next_marker_style(size=6*self.ui_scale)
                implot.plot_scatter('Sequence', xs=xs, ys=ys)
                implot.end_plot()
    
    def draw_toolbar_autoUI(self, containers=None):
        self.state['dim3'].active = self.state.plot3d
        draw_container(self.state, reset_button=True)
        imgui.text(f'Arr: {self.arr_create_time*1000:.0f}ms')

if __name__ == '__main__':
    viewer = Viewer('LDS viewer')
