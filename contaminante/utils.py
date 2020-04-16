import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

import lightkurve as lk
from astropy.stats import sigma_clip, sigma_clipped_stats
import pandas as pd
from tqdm.notebook import tqdm

import warnings

from numpy.linalg import solve
from scipy.sparse import csr_matrix, diags

import astropy.units as u

from .gaia import plot_gaia

def search(targetid, mission, search_func=lk.search_targetpixelfile):
    """Convenience function to help lightkurve searches"""
    if search_func == lk.search_targetpixelfile:
        sr = search_func(targetid, mission=mission)
        numeric = int(''.join([char for char in "KIC {}".format(targetid) if char.isnumeric()]))
        numeric_s = np.asarray([int(''.join([char for char in sr.target_name[idx] if char.isnumeric()])) for idx in range(len(sr))])
        sr = lk.SearchResult(sr.table[numeric_s == numeric])
    elif search_func == lk.search_tesscut:
        sr = search_func(targetid)
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


def build_model(tpf, lc, cbvs=None, t_model=None, errors=False, cadence_mask=None, background=False, spline=True):
    """ Build a model for the pixel level light curve """
    with warnings.catch_warnings():
        # I don't want to fix runtime warnings...
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if cadence_mask is None:
            cadence_mask = np.ones(len(tpf.time)).astype(bool)

        SA = build_X(tpf, lc.flux, t_model=t_model, cbvs=cbvs, spline=spline, background=background)
        model = np.zeros(tpf.flux.shape)
        if errors:
            model_err = np.zeros(tpf.flux.shape)

        if t_model is not None:
            transit_pixels = np.zeros(tpf.flux.shape[1:])
            transit_pixels_err = np.zeros(tpf.flux.shape[1:])

        for idx in (range(tpf.shape[1])):
            for jdx in range(tpf.shape[2]):

                f = tpf.flux[:, idx, jdx]
                fe = tpf.flux_err[:, idx, jdx]
                if not np.isfinite(f).any():
                    continue

                SA_dot_sigma_f_inv = csr_matrix(SA[cadence_mask].multiply(1/fe[cadence_mask, None]**2))
                sigma_w_inv = (SA[cadence_mask].T.dot(SA_dot_sigma_f_inv)).toarray()
                B = (SA[cadence_mask].T.dot((f/fe**2)[cadence_mask]))
                w = solve(sigma_w_inv, B)

                model[:, idx, jdx] = SA.dot(w)
                sigma_w = np.linalg.inv(sigma_w_inv)

                if t_model is not None:
                    transit_pixels[idx, jdx] = w[-1]
                    transit_pixels_err[idx, jdx] = np.std([np.random.multivariate_normal(w, sigma_w)[-1] for count in np.arange(50)])

                if errors:
                    samples = np.asarray([np.dot(SA, np.random.multivariate_normal(w, sigma_w)) for count in np.arange(100)]).T
                    model_err[:, idx, jdx] = np.median(samples, axis=1) - np.percentile(samples, 16, axis=1)

        if t_model is not None:
            if errors:
                return model, model_err, transit_pixels, transit_pixels_err
            return model, transit_pixels, transit_pixels_err
        if errors:
            return model, model_err
        return model

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


def get_centroid_plot(targetid, period, t0, duration, mission='kepler', gaia=False):
    """ Plot where the centroids of the transiting target are in a TPF"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        background = False
        sr = search(targetid, mission)
        if (mission.lower() == 'tess') and (len(sr) == 0):
            tpfs = search(targetid, period, lk.search_tesscut).download_all(cutout_size=(10, 10))
            background = True
        elif len(sr) == 0:
            raise ValueError('No target pixel files exist for {} from {}'.format(targetid, mission))
        else:
            tpfs = sr.download_all()
        contaminator = None
        target = None
        coords_weights, coords_ra, coords_dec, ra_target, dec_target = [], [], [], [], []

        for tpf in tqdm(tpfs, desc='Modeling TPFs'):
            tpf = tpf[(np.nansum(tpf.flux, axis=(1, 2)) != 0) & (np.nansum(tpf.flux_err, axis=(1, 2)) != 0)]
            aper = tpf.pipeline_mask
            if not (aper.any()):
                aper = tpf.create_threshold_mask()
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
            ra_target.append(np.average(tpf.get_coordinates()[0][:, aper].mean(axis=0), weights=np.nanmedian(tpf.flux, axis=0)[aper]**0.5))
            dec_target.append(np.average(tpf.get_coordinates()[1][:, aper].mean(axis=0), weights=np.nanmedian(tpf.flux, axis=0)[aper]**0.5))

            if (mission.lower() == 'kepler') | (mission.lower() == 'k2'):
                cbvs = lk.correctors.KeplerCBVCorrector(lc).cbv_array[:2].T
            else:
                cbvs = None

            t_model = bls.get_transit_model(period=period, transit_time=t0, duration=duration).flux
            t_model -= np.nanmedian(t_model)
            t_model /= bls.depth[0]
            if t_model.sum() == 0:
                continue

#            cbvs = None
            model2, transit_pixels, transit_pixels_err = build_model(tpf, lc, cbvs=cbvs, t_model=t_model, background=background)

            contaminant_aper = (transit_pixels/transit_pixels_err) > 5

            # If all the "transit" pixels are contained in the aperture, continue.
            if np.in1d(np.where(contaminant_aper.ravel()), np.where(aper.ravel())).all():
                continue

            coords = np.asarray(tpf.get_coordinates())[:, :, contaminant_aper].mean(axis=1)
            coords_ra.append(coords[0])
            coords_dec.append(coords[1])
            coords_weights.append((transit_pixels/transit_pixels_err)[contaminant_aper])
            if contaminant_aper.any():
                contaminated_lc = tpf.to_lightcurve(aperture_mask=contaminant_aper)
                if contaminator is None:
                    contaminator = build_lc(tpf, contaminant_aper, background=background, cadence_mask=t_mask, spline_period=period * 4)#contaminated_lc#.flatten(window_length)
                else:
                    contaminator = contaminator.append(build_lc(tpf, contaminant_aper, background=background, cadence_mask=t_mask, spline_period=period * 4))#contaminated_lc.flatten(window_length))

        ra_target, dec_target = np.mean(ra_target), np.mean(dec_target)

        if len(coords_ra) != 0:
            ra = np.average(np.hstack(coords_ra), weights=np.hstack(coords_weights))
            dec = np.average(np.hstack(coords_dec), weights=np.hstack(coords_weights))
        else:
            ra, dec = np.nan, np.nan


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
        ax.scatter(ra_target, dec_target, c='g', marker='x', label='Target', s=100)
        ax.scatter(ra, dec, c='r', marker='x', label='Source Of Transit', s=100)
        ax.legend(frameon=True)
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
        bin_points = np.max([2, int(len(lc.time)/((lc.time[-1] - lc.time[0])/0.5*period))])
        target.fold(period, t0).bin(bin_points, method='median').errorbar(c='g', label="Target", ax=ax, marker='.')
        if contaminator is not None:
            contaminator.fold(period, t0).bin(bin_points, method='median').errorbar(ax=ax, c='r', marker='.', label="Source of Transit")
        return fig, target
