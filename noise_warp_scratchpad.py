import torch
import rp
def noise_warp_wxyc(I, F):


	#Input assertions
	assert F.device==I.device
	assert F.ndim==2, 'F stands for flow, and its in [x y]·h·w form'
	assert I.ndim==3, 'I stands for input, in [ω x y c]·h·w form where ω=weights, x and y are offsets, and c is num noise channels'
	xyωc, h, w = I.shape
	assert F.shape==(2,h,w) # Should be [x y]·h·w
	device=I.device
	
	#How I'm going to address the different channels:
	x   = 0        # 
	y   = 1        # 
	xy  = 2        # I[:xy]
	xyω = 3        # I[:xyω]
	ω   = 2        # I[ω]
	c   = xyωc-xyω # I[c:]
	ωc  = xy       # I[ωc:]
	assert c, 'I has no noise channels. There is nothing to warp.'

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
	pre_expand[ωc:] = rp.torch_remap_image(I[ωc:], * -F.round(), relative=True, interp="nearest")

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


	









	








	

