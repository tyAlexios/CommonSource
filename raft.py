import rp
import torch
import torchvision.transforms
from torchvision.models.optical_flow import raft_large, raft_small

class RaftOpticalFlow(rp.CachedInstances):
    def __init__(self, device, version='large'):

        models = {
            'large' : raft_large,
            'small' : raft_small,
        }
        assert version in models
        model = models[version]
        model = model(pretrained=True, progress=False).to(device)
        model.eval()

        self.version = version
        self.device = device
        self.model = model

    def _preprocess_image(self, image):
        assert rp.is_image(image)
        
        image = rp.as_float_image(rp.as_rgb_image(image))

        #Floor height and width to the nearest multpiple of 8
        height, width = rp.get_image_dimensions(image)
        new_height = (height // 8) * 8
        new_width  = (width  // 8) * 8

        T = torchvision.transforms
        transforms = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
                T.Resize(size=(new_height, new_width)),
            ]
        )
        
        output = transforms(image)[None].to(self.device).float()

        assert rp.is_torch_tensor(output)
        assert output.shape == (1, 3, new_height, new_width)

        return output
    
    def __call__(self, from_image, to_image):
        assert rp.is_image(from_image)
        assert rp.is_image(to_image)
        assert rp.get_image_dimensions(from_image) == rp.get_image_dimensions(to_image)
        
        height, width = rp.get_image_dimensions(from_image)
        
        with torch.no_grad():
            img1 = self._preprocess_image(from_image)
            img2 = self._preprocess_image(to_image  )
            
            list_of_flows = self.model(img1, img2)
            output_flow = list_of_flows[-1][0]
    
            # Resize the predicted flow back to the original image size
            resize = torchvision.transforms.Resize((height, width))
            output_flow = resize(output_flow[None])[0]

        assert rp.is_torch_tensor(output_flow)
        assert output_flow.shape == (2, height, width)

        return output_flow
