import numpy as np

def clip(array, clip_tr=0.01):
    """
    Clip array values to [clip_tr, 1-clip_tr]
    
    Input
        array: np.array
        clip_tr: threshold for clipping propensity scores
    
    Output
        array: np.array
    """
    
    if clip_tr is not None:
        lv_idx = array<clip_tr
        array[lv_idx] = clip_tr
        hv_idx = array>1-clip_tr
        array[hv_idx] = 1-clip_tr
    return array