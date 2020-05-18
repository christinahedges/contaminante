<center><img width = "900" src="https://github.com/christinahedges/contaminante/blob/master/docs/figures/logo.png?raw=true"/></center>

# contaminante
*kohn - tah - mee - nahn - teh*

A package to help find the contaminant transiting source in NASA's *Kepler*, *K2* or *TESS* data. When hunting for transiting planets, sometimes signals come from neighboring contaminants. This package helps users identify where the transiting signal comes from in their data.

## What does `contaminante` do?

`contaminante` uses pixel level modeling of the *TargetPixelFile* data from NASA's astrophysics missions that are processed with the *Kepler* pipeline. The output of `contaminante` is a Python dictionary containing the source location and transit depth, and a contaminant location and depth. Optionally you can output a figure showing

1.  Where the main target is centered in all available TPFs.
2.  What the phase curve looks like for the main target
3.  Where the transiting source is centered in all available TPFs, if a transiting source is located outside the main target
4.  The transiting source phase curve, if a transiting source is located outside the main target

An example output is shown below for a target with a transiting contaminant.

<center><img width = "900" src="https://github.com/christinahedges/contaminante/blob/master/docs/figures/FP.png?raw=true"/></center>

Where as a transit that is centered on the target gives the following output:

<center><img width = "200" src="https://github.com/christinahedges/contaminante/blob/master/docs/figures/real.png?raw=true"/></center>

## How to use `contaminante`

You can check out our [tutorial](https://github.com/christinahedges/contaminante/blob/master/tutorials/tutorial-1_how-to-use-contaminante.ipynb) for how to run `contaminante`. To run `contaminante` you will need a target name, a transit period, a transit center and a transit duration.

## Installation

You can install `contaminante` using pip:

```
pip install contaminate --upgrade
```

You can also install `contaminante` by cloning this repo:

```
git clone https://github.com/christinahedges/contaminante
cd contaminante
python setup.py install
```

### Help, I can't install `contaminante`

You might not be able to install `contaminante` because your computer doesn't support some of the features, or perhaps you're new to Python. Don't worry, you can still use `contaminante`! If you're struggling to install, try running `contaminante` online using Google's Colaboratory. You can click [here](broken) to open a new Colaboratory notebook and run `contaminante` in the cloud!

## Dependencies

`contaminante` uses the most up to date version of [`lightkurve`](https://github.com/keplerGO/lightkurve), and uses some of the features available in v2.0. Make sure your `lightkurve` installation is up to date before using contaminante.

## Acknowledgements
