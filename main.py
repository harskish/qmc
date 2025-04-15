from collections import defaultdict
import numpy as np
from math import sqrt
from functools import lru_cache
import matplotlib.pyplot as plt
import glfw
import pyviewer
from pyviewer.toolbar_viewer import AutoUIViewer
from pyviewer.params import *
from imgui_bundle import implot

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
# - Shufflig (e.g. Owen): works well
# - Random sample offsetting: might be slow, depending on sampler
# - Cranley-patterson rotation: increases variance

BIG_PRIME = 15487313

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

# extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
def R2(i: int, dim: int, N: int, seed: int = 0):
    max_dim = 2 # looks bad for higher values
    p = pow(1/phi(max_dim), dim+1) % 1 # inverse power of plastic number
    return cranley_patterson_rotation(p*i, dim, seed)

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

def sobol_owen_cpp(*args, **kwargs):
    return sobol_cpp_impl('sobol_owen', *args, **kwargs)

def sobol_rds_cpp(*args, **kwargs):
    return sobol_cpp_impl('sobol_rds', *args, **kwargs)

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

def plot_2d(func, dim1, dim2, *args, N=1024, **kwargs):
    global fig
    if func.__name__ == 'hammersley':
        kwargs['N'] = N
    xs = [func(i, dim1, *args, **kwargs) for i in range(N)]
    ys = [func(i, dim2, *args, **kwargs) for i in range(N)]
    plt.plot(xs, ys, 'bo')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f'{func.__name__} dims=({dim1},{dim2})\nseed{kwargs.get("seed", 0)}')
    return fig

@strict_dataclass
class State(ParamContainer):
    N: Param = IntParam('Samples', 128, 1, 2048)
    seq: Param = EnumSliderParam('Sequence', R2, [
        murmur, sobol, sobol_cp, sobol_rds,
        sobol_owen, cascaded_sobol,
        hammersley, halton, leaped_halton, R2,
    ], lambda f: f.__name__)
    seed: Param = IntParam('Seed', 0, 0, 99, buttons=True)
    dim1: Param = IntParam('Dimension 1', 0, 0, 50, buttons=True)
    dim2: Param = IntParam('Dimension 2', 1, 0, 50, buttons=True)
    
class Viewer(AutoUIViewer):
    def setup_state(self):
        self.state = State()
    
    def draw_pre(self):
        state = self.state
        W, H = glfw.get_window_size(self.v._window)
        style = imgui.get_style()
        avail_h = H - self.menu_bar_height - 2*style.window_padding.y
        avail_w = W - self.toolbar_width
        plot_side = min(avail_h, avail_w)
        xs = np.array([state.seq(i, state.dim1, seed=state.seed, N=state.N) for i in range(state.N)])
        ys = np.array([state.seq(i, state.dim2, seed=state.seed, N=state.N) for i in range(state.N)])
        implot.set_next_axes_limits(0, 1, 0, 1)
        implot.set_next_marker_style(size=6*self.ui_scale)
        if implot.begin_plot('LDS', size=(plot_side, plot_side)):
            implot.plot_scatter('Sequence', xs=xs, ys=ys)
            implot.end_plot()

if __name__ == '__main__':
    viewer = Viewer('LDS viewer')
