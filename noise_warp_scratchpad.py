import torch
import rp
from .noise_warp import *


def xy_meshgrid_like_image(image):
    """
    Example:
        >>> image=load_image('https://picsum.photos/id/28/367/267')
        ... image=as_torch_image(image)
        ... xy=xy_meshgrid_like_image(image)
        ... display_image(full_range(as_numpy_array(xy[0])))
        ... display_image(full_range(as_numpy_array(xy[1])))
    """
    assert image.ndim == 3, "image is in CHW form"
    c, h, w = image.shape

    y, x = torch.meshgrid(
        torch.arange(h),
        torch.arange(w),
    )

    output = torch.stack(
        [x, y],
    ).to(image.device, image.dtype)

    assert output.shape == (2, h, w)
    return output


def noise_to_xyωc(noise):
    assert noise.ndim == 3, "noise is in CHW form"
    zeros=torch.zeros_like(noise[0])
    ones =torch.ones_like (noise[0])

    #Prepend [dx=0, dy=0, weights=1] channels
    output=torch.concat([zeros, zeros, ones, noise])
    return output

def xyωc_to_noise(xyωc):
    assert xyωc.ndim == 3, "xyωc is in [ω x y c]·h·w form"
    assert xyωc.shape[0]>3, 'xyωc should have at least one noise channel'
    noise=xyωc[3:]
    return noise

def noise_warp_xyωc(I, F):
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

    #Compute the grid of xy indices
    grid = xy_meshgrid_like_image(I)
    assert grid.shape==(2,h,w) # Shape is [x y]·h·w

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
    pos = (grid + pre_shrink).round()
    in_bounds = (0<= pos[x] < w) & (0<= pos[y] < w)
    in_bounds = in_bounds[None] #Make the matrix an image tensor
    out_of_bounds = ~in_bounds
    assert out_of_bounds.shape==(1,h,w)
    pre_shrink[ out_of_bounds ] = init

    #Deal with shrink positions offsets
    scat_xy = pre_shrink[:xy].round()
    pre_shrink[:xy] -= scat_xy
    scat = lambda tensor: rp.torch_scatter_add_image(tensor, *scat_xy, relative=True)

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

    torch.where()