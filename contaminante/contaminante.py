"""Basic contaminante functionality"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from scipy.ndimage import label

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

import lightkurve as lk
from lightkurve import TargetPixelFileCollection as TPFC
from lightkurve.targetpixelfile import TargetPixelFile as TPF
from lightkurve.correctors.designmatrix import create_sparse_spline_matrix


import astropy.units as u

from scipy.sparse import csr_matrix, diags, hstack
from astropy.timeseries import BoxLeastSquares


def _label(tpf):
    if hasattr(tpf, "quarter"):
        return f"{tpf.to_lightcurve().label}, Quarter {tpf.quarter}"
    elif hasattr(tpf, "campaign"):
        return f"{tpf.to_lightcurve().label}, Campaign {tpf.campaign}"
    elif hasattr(tpf, "sector"):
        return f"{tpf.to_lightcurve().label}, Sector {tpf.sector}"
    else:
        return "{tpf.to_lightcurve().label}"


#
# class TargetPixelFileCollection(TPFC):
#     def __init__(self, tpfs):
#         """This is an instance of a `lk.TargetPixelFileCollection`, with the additional
#         method `calculate_contamination`. It's only purpose is to add the functionality to calculate contamination."""
#         if isinstance(tpfs, (list, TPFC)):
#             super().__init__(tpfs)
#         elif isinstance(tpfs, TPF):
#             super().__init__([tpfs])
#         else:
#             raise ValueError(
#                 "please pass `lk.TargetPixelFile` or `lk.TargetPixelFileCollection`"
#             )
#
#         # remove nans
#         for idx, tpf in enumerate(self):
#             aper = tpf.pipeline_mask
#             if not (aper.any()):
#                 aper = tpf.create_threshold_mask()
#             mask = (np.abs((tpf.pos_corr1)) < 10) & ((np.gradient(tpf.pos_corr2)) < 10)
#             mask &= np.nan_to_num(tpf.to_lightcurve(aperture_mask=aper).flux) != 0
#             self[idx] = self[idx][mask]


def calculate_contamination(
    tpfs, period, t0, duration, sigma=5, plot=True, cbvs=True, **kwargs
):
    """Calculate the contamination for a target
    Parameters
    ----------
    period : float
        Period of transiting object in days
    t0 : float
        Transit midpoint of transiting object in days
    duration : float
        Duration of transit in days
    sigma : float
        The significance level at which to create an aperture for the contaminanting source.
        If the apertures are large, try increasing sigma. If the apertures are small,
        or contaminante fails, you could try (slightly) lowering sigma.
    plot: bool
        If True, will generate a figure
    cbvs : bool
        If True, will use Kepler/TESS CBVs to detrend. Default is True
    sff : bool
        If True, will use the SFF method to correct variability. Default is False.
    spline_period : float
        The period of a spline to fit. For short period variability,
        set this value to a smaller number. Default is 0.75 days.


    Returns
    -------
    result : list of dict
        List of dictionaries containing the contamination properties
        If plot is True, will show a figure, and will put the
        matplotlib.pyplot.figure object into the result dictionary.
    """

    if isinstance(tpfs, (list)):
        tpfs = TPFC(tpfs)
    elif isinstance(tpfs, TPF):
        tpfs = TPFC([tpfs])
    elif not isinstance(tpfs, TPFC):
        raise ValueError(
            "please pass `lk.TargetPixelFile` or `lk.TargetPixelFileCollection`"
        )

    # remove nans
    for idx, tpf in enumerate(tpfs):
        aper = tpf.pipeline_mask
        if not (aper.any()):
            aper = tpf.create_threshold_mask()
        mask = (np.abs((tpf.pos_corr1)) < 10) & ((np.gradient(tpf.pos_corr2)) < 10)
        mask &= np.nan_to_num(tpf.to_lightcurve(aperture_mask=aper).flux) != 0
        tpfs[idx] = tpfs[idx][mask]

    results = []
    for tpf in tqdm(tpfs, desc="Modeling TPFs"):

        aper = tpf.pipeline_mask
        if not (aper.any()):
            aper = tpf.create_threshold_mask()

        lc = tpf.to_lightcurve(aperture_mask=aper).normalize()
        bls = lc.to_periodogram("bls", period=[period, period], duration=duration)
        t_mask = bls.get_transit_mask(period=period, transit_time=t0, duration=duration)
        # Correct light curve
        if cbvs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cbv_array = (
                    lk.correctors.CBVCorrector(
                        tpf.to_lightcurve(aperture_mask=aper),
                        interpolate_cbvs=True,
                        extrapolate_cbvs=True,
                    )
                    .cbvs[0]
                    .to_designmatrix()
                    .X[:, :4]
                )
        else:
            cbv_array = None

        r1, c1 = tpf.estimate_centroids(aperture_mask=aper)
        r1 -= np.median(r1)
        c1 -= np.median(c1)
        X = build_X(lc.time.jd, r1.value, c1.value, cbvs=cbv_array, **kwargs)

        X = X[:, np.asarray(X.sum(axis=0))[0] != 0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dm1 = lk.SparseDesignMatrix(
                X,
                name="X",
                prior_mu=np.hstack([np.zeros(X.shape[1] - 1), 1]),
                prior_sigma=np.hstack([np.ones(X.shape[1] - 1) * 1e2, 0.1]),
            )
        r = lk.RegressionCorrector(lc.copy())
        target = r.correct(dm1, cadence_mask=~t_mask)
        stellar_lc = r.diagnostic_lightcurves["X"].flux
        # Find a transit model
        bls = target.to_periodogram("bls", period=[period, period], duration=duration)
        t_model = (
            ~bls.get_transit_mask(period=period, transit_time=t0, duration=duration)
        ).astype(float) - 1
        depth = bls.compute_stats(period=period, transit_time=t0, duration=duration)[
            "depth"
        ][0]

        X = build_X(
            lc.time.jd,
            r1.value,
            c1.value,
            flux=stellar_lc,
            t_model=t_model,
            cbvs=cbv_array,
            **kwargs,
        )
        X = X[:, np.asarray(X.sum(axis=0))[0] != 0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dm = lk.SparseDesignMatrix(
                X,
                name="X",
                prior_mu=np.hstack([np.zeros(X.shape[1] - 2), 1, depth]),
                prior_sigma=np.hstack([np.ones(X.shape[1] - 2) * 1e4, 0.1, 0.1]),
            )

        model = np.zeros(tpf.flux.shape)
        model_err = np.zeros(tpf.flux.shape)

        # Hard coded saturation limit. Probably not ideal.
        saturated = np.max(np.nan_to_num(tpf.flux.value), axis=0) > 1.4e5
        if saturated.any():
            dsat = np.gradient(saturated.astype(float), axis=0)
            if (~np.any(dsat == 0.5)) | (~np.any(dsat == -0.5)):
                raise ValueError(
                    "Too close to a saturation column that isn't fully captured."
                )
            saturated |= np.abs(dsat) != 0
        pixels = tpf.flux.value.copy()
        pixels_err = tpf.flux_err.value.copy()

        transit_pixels = np.zeros(tpf.flux.shape[1:])
        transit_pixels_err = np.zeros(tpf.flux.shape[1:])

        for jdx, s in enumerate(saturated.T):
            if any(s):
                l = np.where(s)[0][s.sum() // 2]
                pixels[:, s, jdx] = np.nan
                pixels[:, l, jdx] = tpf.flux.value[:, s, jdx].sum(axis=(1))
                pixels_err[:, l, jdx] = (
                    (tpf.flux_err.value[:, s, jdx] ** 2).sum(axis=(1)) ** 0.5
                ) / s.sum()

        for idx in range(tpf.shape[1]):
            for jdx in range(tpf.shape[2]):
                if np.nansum(pixels[:, idx, jdx]) == 0:
                    continue
                r.lc.flux = pixels[:, idx, jdx] / np.median(pixels[:, idx, jdx])
                r.lc.flux_err = pixels_err[:, idx, jdx] / np.median(pixels[:, idx, jdx])

                r.correct(dm)
                transit_pixels[idx, jdx] = r.coefficients[-1]
                sigma_w_inv = X.T.dot(X / r.lc.flux_err[:, None] ** 2) + np.diag(
                    1 / dm.prior_sigma ** 2
                )
                transit_pixels_err[idx, jdx] = (
                    np.asarray(np.linalg.inv(sigma_w_inv)).diagonal()[-1] ** 0.5
                )

        for jdx, s in enumerate(saturated.T):
            if any(s):
                l = np.where(s)[0][s.sum() // 2]
                transit_pixels[s, jdx] = transit_pixels[l, jdx]
                transit_pixels_err[s, jdx] = transit_pixels_err[l, jdx]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            contaminant_aper = create_threshold_mask(
                transit_pixels / transit_pixels_err, sigma
            )
        contaminated_lc = tpf.to_lightcurve(aperture_mask=contaminant_aper).normalize()
        r.lc = contaminated_lc
        contaminator = r.correct(dm1, cadence_mask=~t_mask)

        results.append(
            _package_results(
                tpf,
                target=target,
                contaminator=contaminator,
                aper=aper,
                contaminant_aper=contaminant_aper,
                transit_pixels=transit_pixels,
                transit_pixels_err=transit_pixels_err,
                period=period,
                t0=t0,
                duration=duration,
                plot=plot,
            )
        )
    return results


def _package_results(
    tpf,
    target,
    contaminator,
    aper,
    contaminant_aper,
    transit_pixels,
    transit_pixels_err,
    period,
    t0,
    duration,
    plot=False,
):
    """Helper function for packaging up results"""

    # def get_coords(thumb, err, aper, count=400):
    #     Y, X = np.mgrid[: tpf.shape[1], : tpf.shape[2]]
    #     cxs, cys = [], []
    #     for count in range(count):
    #         err1 = np.random.normal(0, err[aper])
    #         cxs.append(np.average(X[aper], weights=thumb[aper] + err1))
    #         cys.append(np.average(Y[aper], weights=thumb[aper] + err1))
    #     cxs, cys = np.asarray(cxs), np.asarray(cys)
    #     cras, cdecs = tpf.wcs.wcs_pix2world(np.asarray([cxs, cys]).T, 1).T
    #     return cras, cdecs

    def get_coords(thumb, err, aper=None):
        if aper is None:
            aper = np.ones(tpf.flux.shape[1:], bool)
        with np.errstate(divide="ignore"):
            Y, X = np.mgrid[: tpf.shape[1], : tpf.shape[2]]
            aper = create_threshold_mask(thumb / err, 3) & aper
            cxs, cys = [], []
            for count in range(500):
                w = np.random.normal(loc=thumb[aper], scale=err[aper])
                cxs.append(np.average(X[aper], weights=w))
                cys.append(np.average(Y[aper], weights=w))
            cxs, cys = np.asarray(cxs), np.asarray(cys)
            k = (cxs > 0) & (cxs < tpf.shape[2]) & (cys > 0) & (cys < tpf.shape[1])
            cxs, cys = cxs[k], cys[k]
            cras, cdecs = tpf.wcs.all_pix2world(np.asarray([cxs, cys]).T, 1).T
        return cras, cdecs

    thumb = np.nanmean(np.nan_to_num(tpf.flux.value), axis=0)
    err = (np.sum(np.nan_to_num(tpf.flux_err.value) ** 2, axis=0) ** 0.5) / len(
        tpf.time
    )
    ra_target, dec_target = get_coords(thumb, err, aper=aper)
    bls = BoxLeastSquares(target.time, target.flux, target.flux_err)
    depths = []
    for i in range(50):
        bls.y = target.flux + np.random.normal(0, target.flux_err)
        depths.append(bls.power(period, duration)["depth"][0])
    target_depth = (np.mean(depths), np.std(depths))

    res = {"target_depth": target_depth}
    res["target_ra"] = np.median(ra_target), np.std(ra_target)
    res["target_dec"] = np.median(dec_target), np.std(dec_target)
    res["target_lc"] = target
    res["target_aper"] = aper

    if contaminant_aper.any():

        ra_contaminant, dec_contaminant = get_coords(transit_pixels, transit_pixels_err)
        bls = BoxLeastSquares(
            contaminator.time, contaminator.flux, contaminator.flux_err
        )
        depths = []
        for i in range(50):
            bls.y = contaminator.flux + np.random.normal(0, contaminator.flux_err)
            depths.append(bls.power(period, duration)["depth"][0])
        contaminator_depth = (np.mean(depths), np.std(depths))
        res["contaminator_ra"] = np.median(ra_contaminant), np.std(ra_contaminant)
        res["contaminator_dec"] = np.median(dec_contaminant), np.std(dec_contaminant)
        res["contaminator_depth"] = contaminator_depth
        res["contaminator_lc"] = contaminator
        res["contaminator_aper"] = contaminant_aper

        d, de = (contaminator_depth[0] - target_depth[0]), np.hypot(
            contaminator_depth[1], target_depth[1]
        )
        res["delta_transit_depth[sigma]"] = d / de

        dra = res["contaminator_ra"][0] - res["target_ra"][0]
        ddec = res["contaminator_dec"][0] - res["target_dec"][0]
        edra = (res["contaminator_ra"][1] ** 2 + res["target_ra"][1] ** 2) ** 0.5
        eddec = (res["contaminator_dec"][1] ** 2 + res["target_dec"][1] ** 2) ** 0.5
        centroid_shift = (((dra ** 2 + ddec ** 2) ** 0.5) * u.deg).to(u.arcsecond)
        ecentroid_shift = (
            centroid_shift * ((2 * edra / dra) ** 2 + (2 * eddec / ddec) ** 2) ** 0.5
        )
        res["centroid_shift"] = (centroid_shift, ecentroid_shift)

    res["period"] = period
    res["t0"] = t0
    res["duration"] = duration
    res["transit_depth"] = transit_pixels
    res["transit_depth_err"] = transit_pixels_err

    if plot:
        res["fig"] = _make_plot(tpf, res)
    return res


def build_X(
    time,
    pos_corr1,
    pos_corr2,
    flux=None,
    t_model=None,
    background=False,
    cbvs=None,
    spline=True,
    spline_period=0.75,
    sff=False,
    windows=20,
    bins=15,
):
    """Build a design matrix to model pixel in target pixel files

    Parameters
    ----------
    tpf : lightkurve.TargetPixelFile
        Input target pixel file to make the design matrix for
    flux : np.ndarray
        The SAP flux to use for creating the design matrix
    t_model: None or np.ndarray
        The transit model, if None no transit model will be used in the design matrix
    cbvs: None or np.ndarray
        Cotrending Basis vectors. If None will not be used in design matrix
    spline: bool
        Whether to use a B-Spline in time
    spline_period: float
        If using a spline, what time period the knots should be spaced at

    Returns
    -------
    SA : scipy.sparse.csr_matrix
        The design matrix to use to detrend the input TPF
    """

    r, c = np.nan_to_num(pos_corr1), np.nan_to_num(pos_corr2)
    r[np.abs(r) > 10] = 0
    c[np.abs(r) > 10] = 0
    breaks = np.where((np.diff(time) > (np.median(np.diff(time)) * 20)))[0] - 1
    breaks = breaks[breaks > 0]
    if sff:
        lc = lk.KeplerLightCurve(
            time=time, flux=time ** 0, centroid_col=pos_corr1, centroid_row=pos_corr2
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = lk.correctors.SFFCorrector(lc)
            _ = s.correct()
            centroids = s.dmc["sff"].X

    else:
        if np.nansum(r) == 0:
            ts0 = np.asarray([np.in1d(time, t) for t in np.array_split(time, breaks)])
            ts1 = np.asarray(
                [
                    np.in1d(time, t) * (time - t.mean()) / (t[-1] - t[0])
                    for t in np.array_split(time, breaks)
                ]
            )
            time_array = np.vstack([ts0, ts1, ts1 ** 2]).T
            centroids = np.copy(time_array)
        else:
            centroids = np.nan_to_num(
                np.vstack(
                    [
                        r ** idx * c ** jdx
                        for idx in np.arange(0, 4)
                        for jdx in range(0, 4)
                    ]
                ).T
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                centroids = lk.correctors.DesignMatrix(centroids).split(list(breaks)).X
    A = csr_matrix(np.copy(centroids))
    if cbvs is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A = hstack(
                [A, lk.DesignMatrix(np.nan_to_num(cbvs)).split(list(breaks)).X]
            ).tocsr()
    if spline:
        spline_dm = create_sparse_spline_matrix(
            time, n_knots=int(np.max([4, int((time[-1] - time[0]) // spline_period)]))
        ).X
        A = hstack([csr_matrix(A), spline_dm]).tocsr()
    if flux is None:
        if t_model is not None:
            return hstack([A, np.atleast_2d(t_model).T]).tocsr()
        return A

    SA = A.multiply(np.atleast_2d(flux).T).tocsr()
    if t_model is not None:
        SA = hstack(
            [SA, np.atleast_2d(np.ones(len(time))).T, np.atleast_2d(t_model).T]
        ).tocsr()
    else:
        SA = hstack([SA, np.atleast_2d(np.ones(len(time))).T]).tocsr()
    return SA


def _make_plot(tpf, res):
    """Helper function for making a plot of contamination results"""
    with plt.style.context("seaborn-white"):
        fig = plt.figure(figsize=(17, 3.5))
        ax = plt.subplot2grid((1, 4), (0, 0))
        ax.set_title(_label(tpf))

        if tpf.mission.lower() == "tess":
            pix = 27 * u.arcsec.to(u.deg)
        elif tpf.mission.lower() in ["kepler", "ktwo", "k2"]:
            pix = 4 * u.arcsec.to(u.deg)
        else:
            pix = 0

        xlim = [1e10, -1e10]
        ylim = [1e10, -1e10]
        ra, dec = np.asarray(np.median(tpf.get_coordinates(), axis=1))
        with np.errstate(divide="ignore"):
            ax.pcolormesh(
                ra,
                dec,
                np.log10(np.nanmedian(np.nan_to_num(tpf.flux.value), axis=0)),
                cmap="Greys_r",
                shading="auto",
            )
        xlim[0] = np.min([np.percentile(ra, 1) - pix, xlim[0]])
        xlim[1] = np.max([np.percentile(ra, 99) + pix, xlim[1]])
        ylim[0] = np.min([np.percentile(dec, 1) - pix, ylim[0]])
        ylim[1] = np.max([np.percentile(dec, 99) + pix, ylim[1]])
        #        import pdb;pdb.set_trace()
        ax.scatter(
            np.hstack(res["target_ra"]),
            np.hstack(res["target_dec"]),
            c="C0",
            marker=".",
            s=10,
            label="Center of Pipeline Aperture",
            zorder=11,
        )
        if "contaminator_ra" in res.keys():
            ax.scatter(
                np.hstack(res["contaminator_ra"]),
                np.hstack(res["contaminator_dec"]),
                c="r",
                marker=".",
                s=13,
                label="Center of Transit Pixels",
                zorder=10,
            )
        ax.legend(frameon=True)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if tpf.mission.lower() == "tess":
            scalebar = AnchoredSizeBar(
                ax.transData,
                27 * u.arcsec.to(u.deg),
                "27 arcsec",
                "lower center",
                pad=0.1,
                color="black",
                frameon=False,
                size_vertical=27 / 100 * u.arcsec.to(u.deg),
            )
        else:
            scalebar = AnchoredSizeBar(
                ax.transData,
                4 * u.arcsec.to(u.deg),
                "4 arcsec",
                "lower center",
                pad=0.1,
                color="black",
                frameon=False,
                size_vertical=4 / 100 * u.arcsec.to(u.deg),
            )

        ax.add_artist(scalebar)
        ax.set_xlabel("RA [deg]")
        ax.set_ylabel("Dec [deg]")

        ax = plt.subplot2grid((1, 4), (0, 1), colspan=3)
        period, t0 = res["period"], res["t0"]
        ax.set_title(_label(tpf) + f" Period: {period}d, T0: {t0}")
        res["target_lc"].fold(period, t0).errorbar(
            c="C0", label="Target", ax=ax, marker=".", markersize=2
        )
        if "contaminator_lc" in res.keys():
            res["contaminator_lc"].fold(period, t0).errorbar(
                ax=ax,
                c="r",
                marker=".",
                label="Source of Transit",
                markersize=2,
            )
    return fig


def create_threshold_mask(thumb, threshold=3, reference_pixel="max"):
    """Lifted from lightkurve.

    Creates a contiguous region where a "thumnbnail" is greater than some threshold
    ----------
    thumb : np.ndarray
        2D image, in this case the transit depth in every pixel divided by the
        error.
    threshold : float
        A value for the number of sigma by which a pixel needs to be
        brighter than the median flux to be included in the aperture mask.
    reference_pixel: (int, int) tuple, 'center', 'max', or None
        (col, row) pixel coordinate closest to the desired region.
        In this case we use the maximum of the thumbnail.
    Returns
    -------
    aperture_mask : ndarray
        2D boolean numpy array containing `True` for pixels above the
        threshold.
    """
    if reference_pixel == "center":
        reference_pixel = (thumb.shape[2] / 2, thumb.shape[1] / 2)
    if reference_pixel == "max":
        reference_pixel = np.where(thumb == np.nanmax(thumb))
        reference_pixel = (reference_pixel[1][0], reference_pixel[0][0])
    vals = thumb[np.isfinite(thumb)].flatten()
    # Create a mask containing the pixels above the threshold flux
    threshold_mask = np.nan_to_num(thumb) >= threshold
    if (reference_pixel is None) or (not threshold_mask.any()):
        # return all regions above threshold
        return threshold_mask
    else:
        # Return only the contiguous region closest to `region`.
        # First, label all the regions:
        labels = label(threshold_mask)[0]
        # For all pixels above threshold, compute distance to reference pixel:
        label_args = np.argwhere(labels > 0)
        distances = [
            np.hypot(crd[0], crd[1])
            for crd in label_args - np.array([reference_pixel[1], reference_pixel[0]])
        ]
        # Which label corresponds to the closest pixel?
        closest_arg = label_args[np.argmin(distances)]
        closest_label = labels[closest_arg[0], closest_arg[1]]
        return labels == closest_label
