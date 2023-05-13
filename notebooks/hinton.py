import matplotlib.pyplot as plt
import numpy as np

def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix.
    
    function for drawing Hinton diagrams as in matplotlib demo:
    https://matplotlib.org/stable/gallery/specialty_plots/hinton_demo.html
    
    As such this function is subject to:
    
    Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved
    
    and is licensed under,
    the conditions set forth in the file:
    "matplotlib_v1.2.0_LICENSE" 
    provided in the same directory as this file.
    """
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log2(np.abs(matrix).max()))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()