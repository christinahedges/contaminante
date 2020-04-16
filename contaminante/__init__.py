#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
# By default Matplotlib is configured to work with a graphical user interface
# which may require an X11 connection (i.e. a display).  When no display is
# available, errors may occur.  In this case, we default to the robust Agg backend.
import platform
if platform.system() == "Linux" and os.environ.get('DISPLAY', '') == '':
    import matplotlib
    matplotlib.use('Agg')

from .version import __version__
from .utils import get_centroid_plot
