"""Tools to query Gaias"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astroquery.vizier import Vizier

Vizier.ROW_LIMIT = -1
from astropy.coordinates import SkyCoord, Distance
from astropy.time import Time
import astropy.units as u

import warnings


def get_gaia(tpf, magnitude_limit=20):
    """Get the gaia sources in a TPF as SkyCoord objects"""
    c1 = SkyCoord(tpf.ra, tpf.dec, frame="icrs", unit="deg")
    result = Vizier.query_region(
        c1,
        catalog=["I/345/gaia2"],
        radius=(np.hypot(*np.asarray(tpf.shape[1:]) / 2) * 4) * u.arcsec,
    )
    result = result[0].to_pandas()
    result = result[result.Gmag < magnitude_limit]
    cs = []
    for idx, d in result.iterrows():
        if d.Plx > 0:
            dist = Distance(parallax=d.Plx * u.milliarcsecond)
        else:
            dist = np.nan * u.parsec

        cs.append(
            (
                SkyCoord(
                    d.RA_ICRS * u.deg,
                    d.DE_ICRS * u.deg,
                    distance=dist,
                    pm_ra_cosdec=d.pmRA * u.milliarcsecond / u.year,
                    pm_dec=d.pmDE * u.milliarcsecond / u.year,
                    obstime=Time("J2015.5"),
                    radial_velocity=np.nanmax([0, d.RV]) * (u.km / u.s),
                ),
                d.Gmag,
            )
        )
    return cs


def plot_gaia(tpfs, ax=None, color="lime", magnitude_limit=20):
    """ Plot the Gaia sources in TPFs, including their space motion. """
    cs = get_gaia(tpfs[0], magnitude_limit=magnitude_limit)
    if ax is None:
        _, ax = plt.subplots()
    label = "Gaia Sources"
    for c in cs:
        ras, decs = [], []
        for t in [tpf.astropy_time[0] for tpf in tpfs]:
            c_prime = c[0].apply_space_motion(t)
            ras.append(c_prime.ra.deg)
            decs.append(c_prime.dec.deg)
        ax.scatter(ras, decs, lw=1, c=color, label=label, zorder=30, marker="x")
        label = ""
    return ax
