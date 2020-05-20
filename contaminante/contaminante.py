"""Basic contaminante functionality"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

import lightkurve as lk

import astropy.units as u

from .gaia import plot_gaia
from .utils import build_X, build_lc, build_model, search

from astropy.timeseries import BoxLeastSquares

def calculate_contamination(targetid, period, t0, duration, mission='kepler', plot=True, gaia=False, quarter=None, sector=None, campaign=None, bin_points=None):
    """Calculate the contamination for a target

    Parameters
    ----------
    targetid : str
        The ID of the target, either KIC, EPIC or TIC from Kepler, K2 or TESS
    period : float
        Period of transiting object in days
    t0 : float
        Transit midpoint of transiting object in days
    duration : float
        Duration of transit in days
    mission : str
        Kepler, K2 or TESS
    plot: bool
        If True, will generate a figure
    gaia: bool
        If True, will plot gaia sources over image
    quarter : int, list or None
        Quarter of Kepler data to use. Specify either using an integer (e.g. `1`) or
        a range (e.g. `[1, 2, 3]`). `None` will return all quarters.
    sector : int, list or None
        Sector of TESS data to use. Specify either using an integer (e.g. `1`) or
        a range (e.g. `[1, 2, 3]`). `None` will return all sectors.
    campaign : int, list or None
        Campaign of K2 data to use. Specify either using an integer (e.g. `1`) or
        a range (e.g. `[1, 2, 3]`). `None` will return all campaigns.
    bin_points : None or int
        Number of points to bin the data by, if None a sensible default will be chosen.
    Returns
    -------
    fig: matplotlib.pyplot.figure
        If plot is True, will return a figure showing the contamination
    result : dict
        Dictionary containing the contamination properties
    """

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
            mask &= np.isfinite(tpf.to_lightcurve(aperture_mask=aper).flux)
            tpf = tpf[mask]
            lc = tpf.to_lightcurve(aperture_mask=aper)

            bls = lc.flatten(21).to_periodogram('bls', period=[period, period])
            t_mask = bls.get_transit_mask(period=period, transit_time=t0, duration=duration)

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
            k = transit_pixels/transit_pixels_err > 3
            xs, ys = [], []
            for count in range(1000):
                err = np.random.normal(0, transit_pixels_err[k])
                xs.append(np.average(X[k], weights=np.nan_to_num(transit_pixels[k] + err)))
                ys.append(np.average(Y[k], weights=np.nan_to_num(transit_pixels[k] + err)))
            xs, ys = np.asarray(xs), np.asarray(ys)
            ras, decs = tpf.wcs.wcs_pix2world(np.asarray([xs + 0.5, ys + 0.5]).T, 1).T
            coords_ra.append(ras)
            coords_dec.append(decs)

            if contaminant_aper.any():
                contaminated_lc = tpf.to_lightcurve(aperture_mask=contaminant_aper)
                if contaminator is None:
                    contaminator = build_lc(tpf, contaminant_aper, cbvs=cbvs, background=background, cadence_mask=t_mask, spline_period=duration * 6)#contaminated_lc#.flatten(window_length)
                else:
                    contaminator = contaminator.append(build_lc(tpf, contaminant_aper, cbvs=cbvs, background=background, cadence_mask=t_mask, spline_period=duration * 6))#contaminated_lc.flatten(window_length))




        bls = BoxLeastSquares(target.time, target.flux, target.flux_err)
        outlier_mask = target.remove_outliers(sigma=10, return_mask=True)[1]
        outlier_mask &= bls.transit_mask(target.time, period, duration, t0)
        target = target[~outlier_mask]
        bls = BoxLeastSquares(target.time, target.flux, target.flux_err)
        depths = []
        for i in range(50):
            bls.y = target.flux + np.random.normal(0, target.flux_err)
            depths.append(bls.power(period, duration)['depth'][0])
        target_depth = (np.mean(depths), np.std(depths))

        res = {'target_depth': target_depth}
        res['target_ra'] = np.hstack(ra_target).mean(), np.hstack(ra_target).std()
        res['target_dec'] = np.hstack(dec_target).mean(), np.hstack(dec_target).std()
        res['target_lc'] = target
        contaminated = False
        if contaminator is not None:
            bls = BoxLeastSquares(contaminator.time, contaminator.flux, contaminator.flux_err)
            outlier_mask = contaminator.remove_outliers(sigma=10, return_mask=True)[1]
            outlier_mask &= bls.transit_mask(contaminator.time, period, duration, t0)
            contaminator = contaminator[~outlier_mask]
            bls = BoxLeastSquares(contaminator.time, contaminator.flux, contaminator.flux_err)

            depths = []
            for i in range(50):
                bls.y = contaminator.flux + np.random.normal(0, contaminator.flux_err)
                depths.append(bls.power(period, duration)['depth'][0])
            contaminator_depth = (np.mean(depths), np.std(depths))

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
                ax.scatter(np.hstack(ra_target), np.hstack(dec_target), c='C0', marker='.', s=2, label='Target', zorder=11, alpha=1/len(tpfs))
                ax.scatter(np.hstack(coords_ra), np.hstack(coords_dec), c='r', marker='.', s=1, label='Source Of Transit', zorder=10, alpha=0.1)
                if gaia:
                    plot_gaia(tpfs, ax=ax)

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
                    target.fold(period, t0).bin(bin_points, method='median').errorbar(c='C0', label="Target", ax=ax, marker='.', markersize=2)
                    if contaminator is not None:
                        contaminator.fold(period, t0).bin(bin_points, method='median').errorbar(ax=ax, c='r', marker='.', label="Source of Transit", markersize=2)
                else:
                    target.fold(period, t0).errorbar(c='C0', label="Target", ax=ax, marker='.', markersize=2)
                    if contaminator is not None:
                        contaminator.fold(period, t0).errorbar(ax=ax, c='r', marker='.', label="Source of Transit", markersize=2)
            return fig, res
    return res
