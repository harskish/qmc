from pathlib import Path
from pyviewer.custom_ops import get_plugin
plugin = get_plugin('burley2020ext',
    ('main.cpp', 'genpoints.cpp', 'faure05.cpp', 'sobol.cpp'), Path(__file__).parent, ldflags=(), cuda=False)

def sample(seq: str, n: int, dim: int, seed: int):
    assert seq in ["random", "sobol", "sobol_rds", "sobol_owen", "laine_karras", "faure05", "faure05_owen"]
    assert 0 <= dim <= 4, 'Only dims [0,4] supported'
    return plugin.sample(seq, n, dim, seed)
