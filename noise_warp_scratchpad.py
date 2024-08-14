import torch
import rp
from .noise_warp import *
def noise_warp_wxyc(I, F):
    #Input assertions
    assert F.device==I.device
    assert F.ndim==2, 'F stands for flow, and its in [x y]·h·w form'
    assert I.ndim==3, 'I stands for input, in [ω x y c]·h·w form where ω=weights, x and y are offsets, and c is num noise channels'
    xyωc, h, w = I.shape
    assert F.shape==(2,h,w) # Should be [x y]·h·w
    device=I.device
    
    #How I'm going to address the different channels:
    x   = 0        #          // index of Δx channel
    y   = 1        #          // index of Δy channel
    xy  = 2        # I[:xy]
    xyω = 3        # I[:xyω]
    ω   = 2        # I[ω]     // index of weight channel
    c   = xyωc-xyω # I[-c:]   // num noise channels
    ωc  = xyωc-xy  # I[-ωc:]
    h_dim = 1
    w_dim = 2
    assert c, 'I has no noise channels. There is nothing to warp.'
    assert (I[ω]>0).all(), 'All weights should be greater than 0'

    #Compute the grid. Todo: cache this.
    grid = torch.stack(torch.meshgrid(torch.arange(h),torch.arange(w))).to(device)
    assert grid.shape==(2,h,w) # Shape is [x y]·h·w

    #The default values we initialize to. Todo: cache this.
    init = torch.empty_like(I)
    init[:xy]=0
    init[ω]=1
    init[c:]=0

    #Caluclate pre-expand
    pre_expand = torch.empty_like(I)
    pre_expand[:xy] = init[:xy]
    pre_expand[-ωc:] = rp.torch_remap_image(I[-ωc:], * -F.round(), relative=True, interp="nearest")

    #Calculate pre-shrink
    pre_shrink = I.copy()
    pre_shrink[:xy] += F
    #Pre-Shrink mask - discard out-of-bounds pixels
    wh = torch.tensor([w,h]).to(device)
    in_bounds  = pre_shrink[:xy] >= 0
    in_bounds &= pre_shrink[:xy] <  wh
    pre_shrink[in_bounds] = init

    #Where mask==True, we output shrink. Where mask==0, we output expand.
    mask = torch.ones(1,h,w,dtype=bool,device=device)
    mask = rp.torch_remap_image(mask, *F.round(), relative=True, interp="nearest")
    assert mask.dtype==torch.bool, 'If this fails we gotta convert it with mask.=astype(bool)'

    ##CONCAT AND REGAUSSIANIZE AND ADJUST WEIGHTS....

    #Horizontally Concat
    concat    = torch.concat([pre_shrink, pre_expand], dim=w_dim)

    #<start> Regaussianize and distribute weights
    //ABSTRACT INTO NOISE/SUM MATRIX
    unique_ωc, counts, index_matrix = unique_pixels(concat[-ωc:])
    unique_c = unique_ωc[:, 1:]
    u = len(unique_ωc)
    assert unique_ωc.shape == (u, 1+c)
    assert unique_ωc.shape == (u,   c)
    assert counts.shape == (u,)
    assert index_matrix.max() == u - 1
    assert index_matrix.min() == 0
    assert index_matrix.shape == (h, w)

    #Regaussianize
    foreign_noise = torch.randn_like(concat[-c:])
    assert foreign_noise.shape == concat[-c:] == (c, h, w)

    summed_foreign_noise_colors = sum_indexed_values(foreign_noise, index_matrix)
    assert summed_foreign_noise_colors.shape == (u, c)

    meaned_foreign_noise_colors = summed_foreign_noise_colors / rearrange(counts, "u -> u 1")
    assert meaned_foreign_noise_colors.shape == (u, c)

    meaned_foreign_noise = indexed_to_image(index_matrix, meaned_foreign_noise_colors)
    assert meaned_foreign_noise.shape == (c, h, w)

    zeroed_foreign_noise = foreign_noise - meaned_foreign_noise
    assert zeroed_foreign_noise.shape == (c, h, w)

    counts_as_colors = rearrange(counts, "u -> u 1")
    counts_image = indexed_to_image(index_matrix, counts_as_colors)
    assert counts_image.shape == (1, h, w)

    #Distribute Weights
    concat[ω] /= counts_image[0]
    # concat[ω] = concat[ω].nan_to_num() #We shouldn't need this, this is a crutch. Final mask should take care of this.

    #To upsample noise, we must first divide by the area then add zero-sum-noise
    concat[-c:] = concat[-c:] / counts_image ** .5
    concat[-c:] = concat[-c:] + zeroed_foreign_noise

    concat[-c:] = concat[-c:] + zeroed_foreign_noise

    #<end> Regaussianize and distribute weights
    


    









    








    

