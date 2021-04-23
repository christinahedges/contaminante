import pytest
import numpy as np
import lightkurve as lk
from .. import contaminante
from astropy.utils.data import get_pkg_data_filename


fname = get_pkg_data_filename("data/test.fits")


def pytest_runtest_setup(item):
    """Our tests will often run in headless virtual environments. For this
    reason, we enforce the use of matplotlib's robust Agg backend, because it
    does not require a graphical display.

    This avoids errors such as:
        c:\hostedtoolcache\windows\python\3.7.5\x64\lib\tkinter\__init__.py:2023: TclError
        This probably means that tk wasn't installed properly.
    """
    import matplotlib

    matplotlib.use("Agg")


def test_contaminante():
    tpfc = lk.read(fname)
    period = 0.700606
    t0 = 131.59767
    duration = 0.993 / 24

    res = contaminante.calculate_contamination(
        tpfc, period, t0, duration, plot=False, cbvs=False
    )
    assert res[0]["delta_transit_depth[sigma]"] > 5
    assert res[0]["centroid_shift"][0].value > 5

    res = contaminante.calculate_contamination(
        tpfc, period, t0, duration, plot=False, cbvs=True
    )
    assert res[0]["delta_transit_depth[sigma]"] > 5
    assert res[0]["centroid_shift"][0].value > 5

    res = contaminante.calculate_contamination(
        tpfc, period, t0, duration, plot=False, cbvs=True, spline=False
    )
    assert res[0]["delta_transit_depth[sigma]"] > 5
    assert res[0]["centroid_shift"][0].value > 5

    res = contaminante.calculate_contamination(
        tpfc, period, t0, duration, plot=False, cbvs=True, sff=True
    )
    assert res[0]["delta_transit_depth[sigma]"] > 5
    assert res[0]["centroid_shift"][0].value > 5

    res = contaminante.calculate_contamination(
        tpfc, 1000, t0, duration, plot=False, cbvs=True, sff=True
    )
    assert (res[0]["transit_depth"] == 0).all()
