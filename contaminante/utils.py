import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import lightkurve as lk
from astropy.stats import sigma_clip, sigma_clipped_stats
import pandas as pd
from tqdm import tqdm

import warnings

from numpy.linalg import solve
from scipy.sparse import csr_matrix, diags

import astropy.units as u

from .gaia import plot_gaia

from scipy.sparse import csr_matrix
from numpy.linalg import solve


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def search(targetid, mission, search_func=lk.search_targetpixelfile, quarter=None, sector=None, campaign=None):
    """Convenience function to help lightkurve searches"""
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
    """Build a design matrix to use in the model"""


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

def build_model(tpf, flux, cbvs=None, t_model=None, errors=False, cadence_mask=None, background=False, spline=True):
    """ Build a model for the pixel level light curve """
    with warnings.catch_warnings():
        # I don't want to fix runtime warnings...
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if cadence_mask is None:
            cadence_mask = np.ones(len(tpf.time)).astype(bool)

        SA = build_X(tpf, flux, t_model=t_model, cbvs=cbvs, spline=spline, background=background, spline_period=10)

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

        for idx in (range(tpf.shape[1])):
            for jdx in range(tpf.shape[2]):

                f = pixels[:, idx, jdx]
                fe = pixels_err[:, idx, jdx]

                fe /= np.nanmean(f)
                f /= np.nanmean(f)

                if not np.isfinite(f).any():
                    continue

                SA_dot_sigma_f_inv = csr_matrix(SA[cadence_mask].multiply(1/fe[cadence_mask, None]**2))
                sigma_w_inv = (SA[cadence_mask].T.dot(SA_dot_sigma_f_inv)).toarray()
                sigma_w_inv += np.diag(1. / prior_sigma**2)

                B = (SA[cadence_mask].T.dot((f/fe**2)[cadence_mask]))
                B += (prior_mu / prior_sigma**2)

                w = solve(sigma_w_inv, B)

                model[:, idx, jdx] = SA.dot(w)
                sigma_w = np.linalg.inv(sigma_w_inv)

                if t_model is not None:
                    samples = np.random.multivariate_normal(w, sigma_w, size=100)[:, -1]
                    transit_pixels[idx, jdx] = np.mean(samples)
                    transit_pixels_err[idx, jdx] = np.std(samples)

                if errors:
                    samp = np.random.multivariate_normal(w, sigma_w, size=100)
                    samples = np.asarray([SA.dot(samp1) for samp1 in samp]).T
                    model_err[:, idx, jdx] = np.median(samples, axis=1) - np.percentile(samples, 16, axis=1)

        #aper = np.copy(transit_pixels/transit_pixels_err)
        #Fix saturated pixels
        for jdx, s in enumerate(saturated.T):
            if any(s):
                l = (np.where(s)[0][s.sum()//2])
                transit_pixels[s, jdx] = transit_pixels[l, jdx]
                transit_pixels_err[s, jdx] = transit_pixels_err[l, jdx]

        aper = transit_pixels/transit_pixels_err > 3

        if t_model is not None:
            if errors:
                return model, model_err, transit_pixels, transit_pixels_err, aper
            return model, transit_pixels, transit_pixels_err, aper
        if errors:
            return model, model_err, aper
        return model, aper


def build_lc(tpf, aperture_mask, cbvs=None, errors=False, cadence_mask=None, background=False, spline=True, spline_period=2):
    """ Build a corrected light curve """
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

        SA_dot_sigma_f_inv = csr_matrix(SA[cadence_mask].multiply(1/fe[cadence_mask, None]**2))
        sigma_w_inv = (SA[cadence_mask].T.dot(SA_dot_sigma_f_inv)).toarray()
        B = (SA[cadence_mask].T.dot((f/fe**2)[cadence_mask]))
        w = solve(sigma_w_inv, B)
        model = SA.dot(w)

        return raw_lc.copy()/model


def calculate_contamination(targetid, period, t0, duration, mission='kepler', plot=True, gaia=False, quarter=None, sector=None, campaign=None, bin_points=None):
    """ Plot where the centroids of the transiting target are in a TPF"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        background = False
        sr = search(targetid, mission, quarter=quarter, sector=sector, campaign=campaign)

        if (mission.lower() == 'tess') and (len(sr) == 0):
            tpfs = search(targetid, mission, lk.search_tesscut, sector=sector).download_all(cutout_size=(10, 10))
            background = True
        elif len(sr) == 0:
            raise ValueError('No target pixel files exist for {} from {}'.format(targetid, mission))
        else:
            tpfs = sr.download_all()
        contaminator = None
        target = None
        coords_weights, coords_weights_err, coords_ra, coords_dec, ra_target, dec_target = [], [], [], [], [], []

        for tpf in tqdm(tpfs, desc='Modeling TPFs'):
            aper = tpf.pipeline_mask
            if not (aper.any()):
                aper = tpf.create_threshold_mask()
            mask = (np.abs((tpf.pos_corr1)) < 10) & ((np.gradient(tpf.pos_corr2)) < 10)
            #mask &= ~tpf.to_lightcurve(aperture_mask=aper).remove_outliers(return_mask=True)[1]
            mask &= np.isfinite(tpf.to_lightcurve(aperture_mask=aper).flux)
            tpf = tpf[mask]
            lc = tpf.to_lightcurve(aperture_mask=aper)

            bls = lc.flatten(21).to_periodogram('bls', period=[period, period])
            t_mask = bls.get_transit_mask(period=period, transit_time=t0, duration=duration)
#            window_length = int(len(lc.time)/((lc.time[-1] - lc.time[0])/(4*period)))
#            window_length = np.min([window_length, len(lc.time)//3])
#            window_length = [(window_length + 1) if ((window_length % 2) == 0) else window_length][0]


            clc = build_lc(tpf, aper, background=background, cadence_mask=t_mask, spline_period=2)
            if target is None:
                target = clc
            else:
                target = target.append(clc)#lc.flatten(window_length))

            if (mission.lower() == 'kepler') | (mission.lower() == 'k2'):
                cbvs = lk.correctors.KeplerCBVCorrector(lc).cbv_array[:2].T
                cbv_dm = lk.DesignMatrix(cbvs, name='cbvs', prior_mu=[1, 1], prior_sigma=[1e2, 1e2]).to_sparse()
                breaks = np.where((np.diff(tpf.time) > (0.0202 * 10)))[0] - 1
                breaks = breaks[breaks > 0]
                breaks = breaks[np.diff(np.append(0, breaks)) > 100]
                cbv_dm.split(list(np.sort(breaks)), inplace=True)
            else:
                cbvs = None


            t_model = bls.get_transit_model(period=period, transit_time=t0, duration=duration)
            if (t_model.flux == np.mean(t_model.flux)).all():
                continue
            fold = t_model.fold(period, t0)
            med = np.median(fold.flux[np.abs(fold.time) > ((duration/period)/2)])
            depth = np.nanmax([0.00001, med - np.nanmedian(fold.flux[np.abs(fold.time) <= ((duration/period)/2)])])
            t_model -= med
            t_model /= depth
            t_model = t_model.flux



            n_knots = int(np.max([4, int(tpf.time[-1] - tpf.time[0]/(duration * 2))]))
            spline_dm = lk.designmatrix.create_sparse_spline_matrix(lc.time, n_knots=n_knots)
            if cbvs is not None:
                dm = lk.SparseDesignMatrixCollection([spline_dm, cbv_dm])
            else:
                dm = spline_dm

            r = lk.RegressionCorrector(lc)
            try:
                clc = r.correct(dm, sigma=2.5, cadence_mask=t_mask)
            except:
                clc = r.correct(dm, sigma=2.5)
            s_lc = r.diagnostic_lightcurves['spline'].normalize()


#            r.diagnose()
#            return
#            cbvs = None
            model, transit_pixels, transit_pixels_err, contaminant_aper = build_model(tpf, s_lc.flux, cbvs=cbvs, t_model=t_model, background=background)


            # # If all the "transit" pixels are contained in the aperture, continue.
            # if np.in1d(np.where(contaminant_aper.ravel()), np.where(aper.ravel())).all():
            #     continue

            if not contaminant_aper.any():
                continue

            thumb = np.nanmean(tpf.flux, axis=0)
            Y, X = np.mgrid[:tpf.shape[1], :tpf.shape[2]]
            k = tpf.pipeline_mask
            cxs, cys = [], []
            for count in range(100):
                err = np.random.normal(0, thumb[k]**0.5)
                cxs.append(np.average(X[k], weights=thumb[k] + err))
                cys.append(np.average(Y[k], weights=thumb[k] + err))
            cxs, cys = np.asarray(cxs), np.asarray(cys)
            cras, cdecs = tpf.wcs.wcs_pix2world(np.asarray([cxs + 0.5, cys + 0.5]).T, 1).T
            ra_target.append(cras)
            dec_target.append(cdecs)

            Y, X = np.mgrid[:tpf.shape[1], :tpf.shape[2]]
            k = transit_pixels/transit_pixels_err > 1
            xs, ys = [], []
            for count in range(1000):
                err = np.random.normal(0, transit_pixels_err[k])
                xs.append(np.average(X[k], weights=np.nan_to_num(transit_pixels[k] + err)))
                ys.append(np.average(Y[k], weights=np.nan_to_num(transit_pixels[k] + err)))
            xs, ys = np.asarray(xs), np.asarray(ys)
            ras, decs = tpf.wcs.wcs_pix2world(np.asarray([xs + 0.5, ys + 0.5]).T, 1).T
            coords_ra.append(ras)
            coords_dec.append(decs)


            # coords = np.asarray(tpf.get_coordinates())[:, :, contaminant_aper].mean(axis=1)
            # coords_ra.append(coords[0])
            # coords_dec.append(coords[1])
            # coords_weights.append(transit_pixels[contaminant_aper])
            # coords_weights_err.append(transit_pixels_err[contaminant_aper])
            if contaminant_aper.any():
                contaminated_lc = tpf.to_lightcurve(aperture_mask=contaminant_aper)
                if contaminator is None:
                    contaminator = build_lc(tpf, contaminant_aper, cbvs=cbvs, background=background, cadence_mask=t_mask, spline_period=duration * 6)#contaminated_lc#.flatten(window_length)
                else:
                    contaminator = contaminator.append(build_lc(tpf, contaminant_aper, cbvs=cbvs, background=background, cadence_mask=t_mask, spline_period=duration * 6))#contaminated_lc.flatten(window_length))

        #ra_target, dec_target = np.mean(ra_target), np.mean(dec_target)

#         if len(coords_ra) != 0:
#             ras, decs = np.zeros(50), np.zeros(50)
#             for count in range(50):
#                 w = np.hstack(coords_weights)
#                 w += np.random.normal(np.zeros(len(w)), np.hstack(coords_weights_err))
#                 ras[count] = np.average(np.hstack(coords_ra), weights=w)
#                 decs[count] = np.average(np.hstack(coords_dec), weights=w)
# #            ra, dec = ras.mean(), decs.mean()
# #            ra_err, dec_err = ras.std(), decs.std()
#         else:
#             ras, decs = np.asarray([np.nan]), np.asarray([np.nan])
# #            ra_err, dec_err = np.nan, np.nan
#
# #        import pdb; pdb.set_trace()
        bls = target.to_periodogram('bls', period=[period, period])
        target_depth = bls.compute_stats(period=period, duration=duration, transit_time=t0)['depth']
        res = {'target_depth': target_depth}
        res['target_ra'] = np.hstack(ra_target).mean(), np.hstack(ra_target).std()
        res['target_dec'] = np.hstack(dec_target).mean(), np.hstack(dec_target).std()
        res['target_lc'] = target
        contaminated = False
        if contaminator is not None:
            bls = contaminator.to_periodogram('bls', period=[period, period])
            contaminator_depth = bls.compute_stats(period=period, duration=duration, transit_time=t0)['depth']
            res['contaminator_depth'] = contaminator_depth
            res['contaminator_ra'] = np.hstack(coords_ra).mean(), np.hstack(coords_ra).std()
            res['contaminator_dec'] = np.hstack(coords_dec).mean(), np.hstack(coords_dec).std()
            res['contaminator_lc'] = contaminator

            d, de = (contaminator_depth[0] - target_depth[0]), np.hypot(contaminator_depth[1], target_depth[1])
            res['delta_transit_depth[sigma]'] = d/de

            if d/de > 8:
                contaminated = True

        res['contaminated'] = contaminated

        if plot:
            with plt.style.context('seaborn-white'):
                fig = plt.figure(figsize=(17, 3.5))
                ax = plt.subplot2grid((1, 4), (0, 0))
                ax.set_title('Target ID: {}'.format(tpfs[0].targetid))

                xlim = [1e10, -1e10]
                ylim = [1e10, -1e10]
                for idx in range(len(tpfs)):
                    ax.pcolormesh(*np.asarray(np.median(tpfs[idx].get_coordinates(), axis=1)), np.log10(np.nanmedian(tpfs[idx].flux, axis=0)), alpha=1/len(tpfs), cmap='Greys_r')
                    xlim[0] = np.min([np.percentile(tpfs[0].get_coordinates()[0], 1), xlim[0]])
                    xlim[1] = np.max([np.percentile(tpfs[0].get_coordinates()[0], 99), xlim[1]])
                    ylim[0] = np.min([np.percentile(tpfs[0].get_coordinates()[1], 1), ylim[0]])
                    ylim[1] = np.max([np.percentile(tpfs[0].get_coordinates()[1], 99), ylim[1]])
            #        import pdb;pdb.set_trace()
                if gaia:
                    plot_gaia(tpfs, ax=ax)
                ax.scatter(np.hstack(ra_target), np.hstack(dec_target), c='g', marker='.', s=0.1, label='Target', zorder=11, alpha=1/len(tpfs))
                ax.scatter(np.hstack(coords_ra), np.hstack(coords_dec), c='r', marker='.', s=0.1, label='Source Of Transit', zorder=10, alpha=0.1)
    #            confidence_ellipse(ras, decs, ax,
    #                alpha=0.5, facecolor='pink', edgecolor='r', zorder=5)
                # confidence_ellipse(np.asarray(ra_target), np.asarray(dec_target), ax,
                #     alpha=0.5, facecolor='lime', edgecolor='g', zorder=5)

                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                if mission.lower() == 'tess':
                    scalebar = AnchoredSizeBar(ax.transData,
                                       27*u.arcsec.to(u.deg), "27 arcsec", 'lower center',
                                       pad=0.1,
                                       color='black',
                                       frameon=False,
                                       size_vertical=27/100*u.arcsec.to(u.deg))
                else:
                    scalebar = AnchoredSizeBar(ax.transData,
                                       4*u.arcsec.to(u.deg), "4 arcsec", 'lower center',
                                       pad=0.1,
                                       color='black',
                                       frameon=False,
                                       size_vertical=4/100*u.arcsec.to(u.deg))

                ax.add_artist(scalebar)
        #        ax.set_aspect('auto')
                ax.set_xlabel('RA [deg]')
                ax.set_ylabel('Dec [deg]')

                ax = plt.subplot2grid((1, 4), (0, 1), colspan=3)
                ax.set_title('Target ID: {}'.format(tpfs[0].targetid))
                if bin_points == None:
                    bin_points = int(((target.time[-1] - target.time[0])/(4*period)))
                if bin_points > 1:
                    target.fold(period, t0).bin(bin_points, method='median').errorbar(c='g', label="Target", ax=ax, marker='.')
                    if contaminator is not None:
                        contaminator.fold(period, t0).bin(bin_points, method='median').errorbar(ax=ax, c='r', marker='.', label="Source of Transit")
                else:
                    target.fold(period, t0).errorbar(c='g', label="Target", ax=ax, marker='.')
                    if contaminator is not None:
                        contaminator.fold(period, t0).errorbar(ax=ax, c='r', marker='.', label="Source of Transit")
        return fig, res
    return res
