""" Utility functions for contaminante """
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

import lightkurve as lk
from astropy.stats import sigma_clip, sigma_clipped_stats

from numpy.linalg import solve
from scipy.sparse import csr_matrix, diags

import astropy.units as u

def search(targetid, mission, search_func=lk.search_targetpixelfile, quarter=None, sector=None, campaign=None):
    """Convenience function to help lightkurve searches

    Parameters
    ----------
    targetid : str
        The ID of the target, either KIC, EPIC or TIC from Kepler, K2 or TESS
    mission : str
        Kepler, K2 or TESS
    search_func : func
        The search function to use, default is `lk.search_targetpixelfile`. Users may
        want `lk.search_tesscut`
    quarter : int, list or None
        Quarter of Kepler data to use. Specify either using an integer (e.g. `1`) or
        a range (e.g. `[1, 2, 3]`). `None` will return all quarters.
    sector : int, list or None
        Sector of TESS data to use. Specify either using an integer (e.g. `1`) or
        a range (e.g. `[1, 2, 3]`). `None` will return all sectors.
    campaign : int, list or None
        Campaign of K2 data to use. Specify either using an integer (e.g. `1`) or
        a range (e.g. `[1, 2, 3]`). `None` will return all campaigns.

    Returns
    -------
    sr : lk.search.SearchResult
        Search result object containing the valid files.
    """
    if search_func == lk.search_targetpixelfile:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if mission.lower() == 'kepler':
                sr = search_func(targetid, mission=mission, quarter=quarter)
            elif (mission.lower() == 'ktwo') | (mission.lower() == 'k2'):
                sr = search_func(targetid, mission=mission, campaign=campaign)
            elif mission.lower() == 'tess':
                sr = search_func(targetid, mission=mission, sector=sector)
            else:
                raise ValueError("No such mission as `'{}'`".format(mission))
            numeric = int(''.join([char for char in "KIC {}".format(targetid) if char.isnumeric()]))
            numeric_s = np.asarray([int(''.join([char for char in sr.target_name[idx] if char.isnumeric()])) for idx in range(len(sr))])
            sr = lk.SearchResult(sr.table[numeric_s == numeric], )
    elif search_func == lk.search_tesscut:
        sr = search_func(targetid, sector=sector)
    else:
        raise ValueError('Search Function is wrong')
    return sr

def build_X(tpf, flux, t_model=None, background=False, cbvs=None, spline=True, spline_period=2, sff=False):
    """Build a design matrix to model pixel in target pixel files

    Parameters
    ----------
    tpf : lightkurve.TargetPixelFile
        Input target pixel file to make the design matrix for
    flux : np.ndarray
        The SAP flux to use for creating the design matrix
    t_model: None or np.ndarray
        The transit model, if None no transit model will be used in the design matrix
    background: None or np.ndarray
        Background model, useful for TESS data. If None will not be used in design matrix
    cbvs: None or np.ndarray
        Cotrending Basis vectors. If None will not be used in design matrix
    spline: bool
        Whether to use a B-Spline in time
    spline_period: float
        If using a spline, what time period the knots should be spaced at
    sff : bool
        Whether to use the SFF method of buildign centroids

    Returns
    -------
    SA : scipy.sparse.csr_matrix
        The design matrix to use to detrend the input TPF
    """

    r, c = np.nan_to_num(tpf.pos_corr1), np.nan_to_num(tpf.pos_corr2)
    r[np.abs(r) > 10] = 0
    c[np.abs(r) > 10] = 0

    if sff:
        r = lk.SFFCorrector(lk.LightCurve(tpf.time, flux))
        _ = r.correct()
        centroids = r.X['sff']
    else:
        breaks = np.where((np.diff(tpf.time) > (0.0202 * 10)))[0] - 1
        breaks = breaks[breaks > 0]

        if r.sum() == 0:
            ts0 = np.asarray([np.in1d(tpf.time, t) for t in np.array_split(tpf.time, breaks)])
            ts1 = np.asarray([np.in1d(tpf.time, t) * (tpf.time - t.mean())/(t[-1] - t[0]) for t in np.array_split(tpf.time, breaks)])
            centroids = np.vstack([ts0, ts1, ts1**2]).T

        else:
            rs0 = np.asarray([np.in1d(tpf.time, t) for t in np.array_split(tpf.time, breaks)])
            rs1 = np.asarray([np.in1d(tpf.time, t) * (r - r[np.in1d(tpf.time, t)].mean()) for t in np.array_split(tpf.time, breaks)])
            cs1 = np.asarray([np.in1d(tpf.time, t) * (c - c[np.in1d(tpf.time, t)].mean()) for t in np.array_split(tpf.time, breaks)])
            centroids = np.vstack([
                                   rs1, cs1, rs1*cs1,
                                   rs1**2, cs1**2, rs1**2*cs1, rs1*cs1**2, rs1**2*cs1**2,
                                   rs1**3*cs1**3, rs1**3*cs1**2, rs1**3*cs1, rs1**3, cs1**3, cs1**3*rs1, cs1**3*rs1**2]).T

    A = np.copy(centroids)
    if cbvs is not None:
        A = np.hstack([A, np.nan_to_num(cbvs)])
    if background:
        bkg = lk.DesignMatrix(tpf.flux[:, ~tpf.create_threshold_mask()]).pca(3).values
        A = np.hstack([A, bkg])
    if spline:
        spline_dm = lk.correctors.designmatrix.create_spline_matrix(tpf.time, n_knots=np.max([4, int((tpf.time[-1] - tpf.time[0])//spline_period)])).values
        A = np.hstack([A, spline_dm])
    SA = np.atleast_2d(flux).T * A
    if t_model is not None:
        SA = np.hstack([SA, np.atleast_2d(np.ones(len(tpf.time))).T, np.atleast_2d(t_model).T])
    else:
        SA = np.hstack([SA, np.atleast_2d(np.ones(len(tpf.time))).T])
    return csr_matrix(SA)

def build_model(tpf, flux, cbvs=None, t_model=None, errors=False, cadence_mask=None, background=False, spline=True, spline_period=2, return_weights=False):
    """ Build a model for the pixel level light curve

        Parameters
        ----------
        tpf : lightkurve.TargetPixelFile
            Input target pixel file to make the design matrix for
        flux : np.ndarray
            The SAP flux to use for creating the design matrix
        cbvs: None or np.ndarray
            Cotrending Basis vectors. If None will not be used in design matrix
        t_model: None or np.ndarray
            The transit model, if None no transit model will be used in the design matrix
        errors: bool
            Whether to return the errors of the models
        cadence_mask: None or np.ndarray
            A mask to specify which cadences to use. Cadences where True will not be used in the analysis.
        background: bool
            Whether to estimate the background flux, useful for TESS
        spline: bool
            Whether to use a B-Spline in time
        spline_period: float
            If using a spline, what time period the knots should be spaced at
        Returns
        -------
        model : np.ndarray
            Model of the TPF, with shape ncadences x npixels x npixels.
        model_err: np.ndarray
            If errors is true, returns model errors
        transit_pixels : np.ndarray
            If t_model is specified, the weight of the transit in each pixel.
            Shape npixel x npixel
        transit_pixels_err : np.ndarray
            If t_model is specified, the error of the weight of the transit in each pixel.
            Shape npixel x npixel
        aper : bool
            The aperture that contains the transit signal (npixels x npixels)
    """

    with warnings.catch_warnings():
        # I don't want to fix runtime warnings...
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if cadence_mask is None:
            cadence_mask = np.ones(len(tpf.time)).astype(bool)

        SA = build_X(tpf, flux, t_model=t_model, cbvs=cbvs, spline=spline, background=background, spline_period=spline_period)

        prior_sigma = np.ones(SA.shape[1]) * 1e-2
        prior_mu = np.zeros(SA.shape[1])
        if t_model is not None:
            prior_mu[-2] = 1
        else:
            prior_mu[-1] = 1


        model = np.zeros(tpf.flux.shape)
        if errors:
            model_err = np.zeros(tpf.flux.shape)

        if t_model is not None:
            transit_pixels = np.zeros(tpf.flux.shape[1:]) * np.nan
            transit_pixels_err = np.zeros(tpf.flux.shape[1:]) * np.nan
            #SA[:, :-1][t_model !=0 ] *= 0


        # Fix Saturation
        saturated = np.max(np.nan_to_num(tpf.flux), axis=0) > 1.4e5
        saturated |= np.abs(np.gradient(saturated.astype(float), axis=0)) != 0
        pixels = tpf.flux.copy()
        pixels_err = tpf.flux_err.copy()

        for jdx, s in enumerate(saturated.T):
            if any(s):
                l = (np.where(s)[0][s.sum()//2])
                pixels[:, s, jdx] = np.nan
                pixels[:, l, jdx] = tpf.flux[:, s, jdx].sum(axis=(1))
                pixels_err[:, l, jdx] = ((tpf.flux_err[:, s, jdx]**2).sum(axis=(1))**0.5)/s.sum()

        ws = np.zeros((tpf.shape[1], tpf.shape[2], SA.shape[1]))
        for idx in (range(tpf.shape[1])):
            for jdx in range(tpf.shape[2]):

                f = pixels[:, idx, jdx]
                if (f < 0).any():
                    continue
                fe = pixels_err[:, idx, jdx]

                fe /= np.nanmean(f)
                f /= np.nanmean(f)

                k = np.nan_to_num(fe) != 0

                if (~k).all():
                    continue

                if not np.isfinite(f).any():
                    continue

                SA_dot_sigma_f_inv = csr_matrix(SA[cadence_mask & k].multiply(1/fe[cadence_mask & k, None]**2))
                sigma_w_inv = (SA[cadence_mask & k].T.dot(SA_dot_sigma_f_inv)).toarray()
                sigma_w_inv += np.diag(1. / prior_sigma**2)

                B = (SA[cadence_mask & k].T.dot((f/fe**2)[cadence_mask & k]))
                B += (prior_mu / prior_sigma**2)

                w = solve(sigma_w_inv, B)
                ws[idx, jdx, :] = w

                model[:, idx, jdx] = SA.dot(w)
                sigma_w = np.linalg.inv(sigma_w_inv)

                if t_model is not None:
                    samples = np.random.multivariate_normal(w, sigma_w, size=100)[:, -1]
                    transit_pixels[idx, jdx] = np.nanmean(samples)
                    transit_pixels_err[idx, jdx] = np.nanstd(samples)

                if errors:
                    samp = np.random.multivariate_normal(w, sigma_w, size=100)
                    samples = np.asarray([SA.dot(samp1) for samp1 in samp]).T
                    model_err[:, idx, jdx] = np.nanmedian(samples, axis=1) - np.nanpercentile(samples, 16, axis=1)

        transit_pixels = np.nan_to_num(transit_pixels)
        #aper = np.copy(transit_pixels/transit_pixels_err)
        #Fix saturated pixels
        for jdx, s in enumerate(saturated.T):
            if any(s):
                l = (np.where(s)[0][s.sum()//2])
                transit_pixels[s, jdx] = transit_pixels[l, jdx]
                transit_pixels_err[s, jdx] = transit_pixels_err[l, jdx]

        aper = transit_pixels/transit_pixels_err > 3

        result = [model]
        if errors:
            result.append(model_err)
        if t_model is not None:
            [result.append(i) for i in [transit_pixels, transit_pixels_err, aper]]
        if return_weights:
            result.append(ws)
        return result

def build_lc(tpf, aperture_mask, cbvs=None, errors=False, cadence_mask=None, background=False, spline=True, spline_period=2):
    """ Build a corrected light curve

        Parameters
        ----------
        tpf : lightkurve.TargetPixelFile
            Input target pixel file to make the design matrix for
        aperture_mask : np.ndarray of bool
            Aperture mask to sum the light curve over
        cbvs: None or np.ndarray
            Cotrending Basis vectors. If None will not be used in design matrix
        errors: bool
            Whether to calculate the errors on the model and propagate them
        cadence_mask: None or np.ndarray
            A mask to specify which cadences to use. Cadences where True will not be used in the analysis.
        background: bool
            Whether to calculate a background model
        spline: bool
            Whether to use a B-Spline in time
        spline_period: float
            If using a spline, what time period the knots should be spaced at

        Returns
        -------
        corrected_lc : lightkurve.LightCurve
            The corrected light curve
        """
    with warnings.catch_warnings():
        # I don't want to fix runtime warnings...
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if cadence_mask is None:
            cadence_mask = np.ones(len(tpf.time)).astype(bool)

        SA = build_X(tpf, np.ones(len(tpf.time)), cbvs=cbvs, spline=spline, background=background, spline_period=spline_period)
        SA = SA[:, :-1]
        raw_lc = tpf.to_lightcurve(aperture_mask=aperture_mask)


        f = raw_lc.flux
        fe = raw_lc.flux_err

        k = np.nan_to_num(fe) != 0

        SA_dot_sigma_f_inv = csr_matrix(SA[cadence_mask & k].multiply(1/fe[cadence_mask & k, None]**2))
        sigma_w_inv = (SA[cadence_mask & k].T.dot(SA_dot_sigma_f_inv)).toarray()
        B = (SA[cadence_mask & k].T.dot((f/fe**2)[cadence_mask & k]))
        w = solve(sigma_w_inv, B)
        model = SA.dot(w)

        return raw_lc.copy()/model
