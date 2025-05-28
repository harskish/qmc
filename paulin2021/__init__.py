from collections import defaultdict
from pathlib import Path
from pyviewer.custom_ops import get_plugin

plugin = get_plugin('paulin2021ext', 'cascadedSobol.cpp', Path(__file__).parent, ldflags=(), cuda=False, unsafe_load_prebuilt=True)
data_dir = (Path(__file__).parent / "data/cascaded_sobol_init_tab.dat").as_posix()

def sample(i: int, dim: int, seed: int, max_pts: int = 1<<30):
    return plugin.sample(i, dim, seed, max_pts, data_dir)

_cache = defaultdict(list)
def sample_ref(i: int, dim: int, seed: int = 0):
    global _cache
    n_dims = 100
    max_pts = 1<<30
    assert i < max_pts and dim < n_dims

    # Non-cached sanity check
    #return plugin.generate(i+1, n_dims, seed, max_pts, data_dir)[i*n_dims+dim]

    key = seed # always generate all available dims
    if len(_cache[key])//n_dims < i+1:
        n_samp = max(i+1, 2*len(_cache[key])//n_dims)
        _cache[key] = plugin.generate(n_samp, n_dims, seed, max_pts, data_dir)
    return _cache[key][i*n_dims+dim] # dims are consecutive
