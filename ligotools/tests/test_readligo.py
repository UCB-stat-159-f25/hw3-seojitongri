import numpy as np
from ligotools import readligo as rl
from pathlib import Path

DATA = Path("data")
fn_H1 = DATA / "H-H1_LOSC_4_V2-1126259446-32.hdf5"

def test_loaddata_output():
    strain, time, chan_dict = rl.loaddata(fn_H1, 'H1')
    assert isinstance(strain, np.ndarray)
    assert isinstance(time, np.ndarray)
    assert isinstance(chan_dict, dict)
    assert len(strain) == len(time)

def test_strain_range():
    strain, time, _ = rl.loaddata(fn_H1, 'H1')
    assert np.all(np.isfinite(strain))
    assert abs(np.mean(strain)) < 1e-18

