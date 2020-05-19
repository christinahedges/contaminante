## How does `contaminante` work?

`contaminante` uses pixel level modeling to model the systematics and astrophysics in each pixel. Each pixel is modeled by components that include:

* A B-spline in time (with knots every 2 days by default)
* A prediction of the centroid position, either using the Kepler Pipieline `POSCORR` values, or building an arclength model similar to that used in the self flat fielding method (see lightkurve.SFFCorrector)
* Optionally, an estimate of the scattered background light (useful for TESS data)
* Optionally, the top Cotrending Basis Vectors from the Kepler pipeline
* A transit model, with period, transit mid point and duration specified by the user.

These components create a design matrix, consisting of predictors of the systematics of the light curve.

In each pixel, `contaminante` finds the best fitting model \\(m\\) for each pixel, where \\(m\\) is given by

\\[ m = S . X . w\\]

where \\(S\\) is an estimate of the astrophysical flux, and \\(X\\) is the design matrix described above. \\(w\\) are the weights of each component. Using L2 regularization, `contaminante` finds the optimum values of \\(w\\) to find the best fitting model \\(m\\) in each pixel. Contaminante then samples to find the uncertainty of each weight \\(\sigma_w\\), assuming Gaussian errors. The weight for the transit model component in each pixel can then be interpretted as the strength of the transiting signal in each pixel. Using the uncertainty, `contaminante` identifies pixels where the transiting signal is measured at a significance >\\(3\sigma\\). These pixels are summed across every quarter, campaign or sector available to find simple aperture photometry of all pixels containing a significant transiting signal. `contaminante` then finds the source center and the original target center, and returns the measured transit depth in each light curve.
