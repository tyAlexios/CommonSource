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
    # assert grid.shape==(2,h,w) # Shape is [x y]·h·w #COMMENTED SO I CAN SEE IF I EVEN USE IT

    #The default values we initialize to. Todo: cache this.
    init = torch.empty_like(I)
    init[:xy]=0
    init[ω]=1
    init[c:]=0

    #Caluclate initial pre-expand
    pre_expand = torch.empty_like(I)
    pre_expand[:xy] = init[:xy]
    pre_expand[-ωc:] = rp.torch_remap_image(I[-ωc:], * -F.round(), relative=True, interp="nearest")

    #Calculate initial pre-shrink
    pre_shrink = I.copy()
    pre_shrink[:xy] += F

    #Pre-Shrink mask - discard out-of-bounds pixels
    wh = torch.tensor([w,h]).to(device)
    in_bounds = 0<= pre_shrink[:xy] < wh
    out_of_bounds = ~in_bounds
    pre_shrink[ out_of_bounds ] = init

    #Deal with shrink positions offsets
    scat_xy = pre_shrink[:xy].round()
    pre_shrink[:xy] -= scat_xy
    scat = lambda tensor: rp.torch_scatter_add_image(tensor, *scat_xy, relative=True, interp='round')

    #Where mask==True, we output shrink. Where mask==0, we output expand.
    shrink_mask = torch.ones(1,h,w,dtype=bool,device=device) #The purpose is to get zeroes where no element is used
    shrink_mask = scat(shrink_mask)
    assert shrink_mask.dtype==torch.bool, 'If this fails we gotta convert it with mask.=astype(bool)'

    #Remove the expansion points where we'll use shrink
    pre_expand[shrink_mask] = init

    #Horizontally Concat
    concat_dim = w_dim
    concat     = torch.concat([pre_shrink, pre_expand], dim=concat_dim)

    #Regaussianize
    concat[-c:], counts_image = regaussianize(concat[-c:])
    assert  counts_image.shape == (1, h, w)

    #Distribute Weights
    concat[ω] /= counts_image[0]
    # concat[ω] = concat[ω].nan_to_num() #We shouldn't need this, this is a crutch. Final mask should take care of this.

    pre_shrink, expand = torch.chunk(concat, chunks=2, dim=concat_dim)
    assert pre_shrink.shape == pre_expand.shape == I.shape
 
    shrink = torch.empty_like(pre_shrink)
    shrink[ω]   = scat(pre_shrink[ω][None])[0]
    shrink[:xy] = scat(pre_shrink[:xy]*pre_shrink[ω][None]) / shrink[ω][None]
    shrink[-c:] = scat(pre_shrink[-c:]*pre_shrink[ω][None]) / scat(pre_shrink[ω][None]**2).sqrt()

    output = torch.where(shrink_mask, shrink, expand)