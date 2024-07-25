import glob
import sys

import fire
import numpy as np
import rp
import torch
from einops import rearrange
from tqdm import tqdm

import raft

sys.path.append(rp.get_path_parent(__file__))


def unique_pixels(image):
    """
    Find unique pixel values in an image tensor and return their RGB values, counts, and inverse indices.

    Args:
        image (torch.Tensor): Image tensor of shape [c, h, w], where c is the number of channels (e.g., 3 for RGB),
                              h is the height, and w is the width of the image.

    Returns:
        tuple: A tuple containing three tensors:
            - unique_colors (torch.Tensor): Tensor of shape [u, c] representing the unique RGB values found in the image,
                                            where u is the number of unique colors.
            - counts (torch.Tensor): Tensor of shape [u] representing the counts of each unique color.
            - index_matrix (torch.Tensor): Tensor of shape [h, w] representing the inverse indices of each pixel,
                                           mapping each pixel to its corresponding unique color index.
    """
    c, h, w = image.shape

    # Rearrange the image tensor from [c, h, w] to [h, w, c] using einops
    pixels = rearrange(image, "c h w -> h w c")

    # Flatten the image tensor to [h*w, c]
    flattened_pixels = rearrange(pixels, "h w c -> (h w) c")

    # Find unique RGB values, counts, and inverse indices
    unique_colors, inverse_indices, counts = torch.unique(flattened_pixels, dim=0, return_inverse=True, return_counts=True)

    # Get the number of unique indices
    u = unique_colors.shape[0]

    # Reshape the inverse indices back to the original image dimensions [h, w] using einops
    index_matrix = rearrange(inverse_indices, "(h w) -> h w", h=h, w=w)

    # Assert the shapes of the output tensors
    assert unique_colors.shape == (u, c)
    assert counts.shape == (u,)
    assert index_matrix.shape == (h, w)
    assert index_matrix.min() == 0
    assert index_matrix.max() == u - 1

    return unique_colors, counts, index_matrix


def sum_indexed_values(image, index_matrix):
    """
    Sum the values in the CHW image tensor based on the indices specified in the HW index matrix.

    Args:
        image (torch.Tensor): Image tensor of shape [C, H, W], where C is the number of channels,
                              H is the height, and W is the width of the image.
        index_matrix (torch.Tensor): Index matrix tensor of shape [H, W] containing indices
                                     specifying the mapping of each pixel to its corresponding
                                     unique value.
                                     Indices range [0, U), where U is the number of unique indices

    Returns:
        torch.Tensor: Tensor of shape [U, C] representing the sum of values in the image tensor
                      based on the indices in the index matrix, where U is the number of unique
                      indices in the index matrix.
    """
    c, h, w = image.shape
    u = index_matrix.max() + 1

    # Rearrange the image tensor from [c, h, w] to [h, w, c] using einops
    pixels = rearrange(image, "c h w -> h w c")

    # Flatten the image tensor to [h*w, c]
    flattened_pixels = rearrange(pixels, "h w c -> (h w) c")

    # Create an output tensor of shape [u, c] initialized with zeros
    output = torch.zeros((u, c), dtype=flattened_pixels.dtype, device=flattened_pixels.device)

    # Scatter sum the flattened pixel values using the index matrix
    output.index_add_(0, index_matrix.view(-1), flattened_pixels)

    # Assert the shapes of the input and output tensors
    assert image.shape == (c, h, w), f"Expected image shape: ({c}, {h}, {w}), but got: {image.shape}"
    assert index_matrix.shape == (h, w), f"Expected index_matrix shape: ({h}, {w}), but got: {index_matrix.shape}"
    assert output.shape == (u, c), f"Expected output shape: ({u}, {c}), but got: {output.shape}"

    return output

def indexed_to_image(index_matrix, unique_colors):
    """
    Create a CHW image tensor from an HW index matrix and a UC unique_colors matrix.

    Args:
        index_matrix (torch.Tensor): Index matrix tensor of shape [H, W] containing indices
                                     specifying the mapping of each pixel to its corresponding
                                     unique color.
        unique_colors (torch.Tensor): Unique colors matrix tensor of shape [U, C] containing
                                      the unique color values, where U is the number of unique
                                      colors and C is the number of channels.

    Returns:
        torch.Tensor: Image tensor of shape [C, H, W] representing the reconstructed image
                      based on the index matrix and unique colors matrix.
    """
    h, w = index_matrix.shape
    u, c = unique_colors.shape

    # Assert the shapes of the input tensors
    assert index_matrix.max() < u, f"Index matrix contains indices ({index_matrix.max()}) greater than the number of unique colors ({u})"

    # Gather the colors based on the index matrix
    flattened_image = unique_colors[index_matrix.view(-1)]

    # Reshape the flattened image to [h, w, c]
    image = rearrange(flattened_image, "(h w) c -> h w c", h=h, w=w)

    # Rearrange the image tensor from [h, w, c] to [c, h, w] using einops
    image = rearrange(image, "h w c -> c h w")

    # Assert the shape of the output tensor
    assert image.shape == (c, h, w), f"Expected image shape: ({c}, {h}, {w}), but got: {image.shape}"

    return image


def demo_pixellation_via_proxy():
    real_image = rp.as_torch_image(
        rp.cv_resize_image(
            rp.load_image("https://i.natgeofe.com/n/4f5aaece-3300-41a4-b2a8-ed2708a0a27c/domestic-dog_thumb_square.jpg"),
            (512, 512),
        )
    )

    c, h, w = real_image.shape

    noise_image = torch.randn(c, h // 4, w // 4)

    # Resize noise_image using nearest-neighbor interpolation to match the dimensions of real_image
    pixelated_noise_image = rp.torch_resize_image(noise_image, 4, "nearest")
    assert pixelated_noise_image.shape==(c,h,w)

    # Find unique pixel values, their indices, and counts in the pixelated noise image
    unique_colors, counts, index_matrix = unique_pixels(pixelated_noise_image)

    # Sum the color values from real_image based on the indices of the unique noise pixels
    summed_colors = sum_indexed_values(real_image, index_matrix)

    # Divide the summed color values by the counts to get the average color for each unique pixel
    average_colors = summed_colors / rearrange(counts, "u -> u 1")

    # Create a new pixelated image using the average colors and the index matrix
    pixelated_dog_image = indexed_to_image(index_matrix, average_colors)

    rp.display_image(pixelated_dog_image)
    
def calculate_wave_pattern(h, w, frame):
    # Create a grid of coordinates
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
    
    # Calculate the distance from the center of the image
    center_x, center_y = w // 2, h // 2
    dist_from_center = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Calculate the angle from the center of the image
    angle_from_center = torch.atan2(y - center_y, x - center_x)
    
    # Calculate the wave pattern based on the distance and angle
    wave_freq = 0.05  # Frequency of the waves
    wave_amp = 10.0   # Amplitude of the waves
    wave_offset = frame * 0.05  # Offset for animation
    
    dx = wave_amp * torch.cos(dist_from_center * wave_freq + angle_from_center + wave_offset)
    dy = wave_amp * torch.sin(dist_from_center * wave_freq + angle_from_center + wave_offset)
    
    return dx, dy

def starfield_zoom(h, w, frame):
    # Create a grid of coordinates
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
    
    # Calculate the distance from the center of the image
    center_x, center_y = w // 2, h // 2
    dist_from_center = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Calculate the angle from the center of the image
    angle_from_center = torch.atan2(y - center_y, x - center_x)
    
    # Calculate the starfield zoom effect
    zoom_speed = 0.01  # Speed of the zoom effect
    zoom_scale = 1.0 + frame * zoom_speed  # Scale factor for the zoom effect
    
    # Calculate the displacement based on the distance and angle
    dx = dist_from_center * torch.cos(angle_from_center) / zoom_scale
    dy = dist_from_center * torch.sin(angle_from_center) / zoom_scale
    
    return dx, dy

def warp_noise(noise, dx, dy, s=4):
    #This is *certainly* imperfect. We need to have particle swarm in addition to this.

    c, h, w = noise.shape
    assert dx.shape==(h,w)
    assert dy.shape==(h,w)

    #s is scaling factor
    hs = h * s
    ws = w * s
    
    #Upscale the warping with linear interpolation. Also scale it appropriately.
    up_dx = rp.torch_resize_image(dx[None], (hs, ws), interp="bilinear")[0]
    up_dy = rp.torch_resize_image(dy[None], (hs, ws), interp="bilinear")[0]
    up_dx *= s
    up_dy *= s

    up_noise = rp.torch_resize_image(noise, (hs, ws), interp="nearest")
    assert up_noise.shape == (c, hs, ws)

    up_noise = rp.torch_remap_image(up_noise, up_dx, up_dy, relative=True, interp="nearest", add_alpha_mask=True)
    up_noise, alpha = up_noise[:-1], up_noise[-1:]
    assert up_noise.shape == (c, hs, ws)
    assert alpha.shape == (1, hs, ws)
    
    # Fill occluded regions with noise...
    fill_noise = torch.randn_like(up_noise)
    up_noise = up_noise * alpha + fill_noise * (1-alpha)
    assert up_noise.shape == (c, hs, ws)

    # Find unique pixel values, their indices, and counts in the pixelated noise image
    unique_colors, counts, index_matrix = unique_pixels(up_noise)
    u = len(unique_colors)
    assert unique_colors.shape == (u, c)
    assert counts.shape == (u,)
    assert index_matrix.max() == u - 1
    assert index_matrix.min() == 0
    assert index_matrix.shape == (hs, ws)

    foreign_noise = torch.randn_like(up_noise)
    assert foreign_noise.shape == up_noise.shape == (c, hs, ws)

    summed_foreign_noise_colors = sum_indexed_values(foreign_noise, index_matrix)
    assert summed_foreign_noise_colors.shape == (u, c)

    meaned_foreign_noise_colors = summed_foreign_noise_colors / rearrange(counts, "u -> u 1")
    assert meaned_foreign_noise_colors.shape == (u, c)

    meaned_foreign_noise = indexed_to_image(index_matrix, meaned_foreign_noise_colors)
    assert meaned_foreign_noise.shape == (c, hs, ws)

    zeroed_foreign_noise = foreign_noise - meaned_foreign_noise
    assert zeroed_foreign_noise.shape == (c, hs, ws)

    counts_as_colors = rearrange(counts, "u -> u 1")
    counts_image = indexed_to_image(index_matrix, counts_as_colors)
    assert counts_image.shape == (1, hs, ws)

    #To upsample noise, we must first divide by the area then add zero-sum-noise
    output = up_noise
    output = output / counts_image ** .5
    output = output + zeroed_foreign_noise

    #Now we resample the noise back down again
    output = rp.torch_resize_image(output, (h, w), interp='area')
    output = output * s #Adjust variance by multiplying by sqrt of area, aka sqrt(s*s)=s

    return output
    
def demo_noise_warp():
    #Run this in a Jupyter notebook and watch the noise go brrrrrrr
    d=rp.JupyterDisplayChannel()
    d.display()
    device='cuda'
    h=w=256
    noise=torch.randn(3,h,w).to(device)
    wdx,wdy=calculate_wave_pattern(h,w,frame=0)
    sdx,sdy=starfield_zoom(h,w,frame=1)

    dx=sdx+2*wdx
    dy=sdy+2*wdy
    
    dx/=dx.max()
    dy/=dy.max()
    Q=-6
    dy*=Q
    dx*=Q
    dx=dx.to(device)
    dy=dy.to(device)
    new_noise=noise

    for _ in range(10000):
        new_noise=warp_noise(new_noise,dx,dy,2)
        # rp.display_image(new_noise)
        d.update(rp.as_numpy_image(new_noise/4+.5))

def get_noise_from_video(
    video_path: str,
    noise_channels: int = 3,
    output_folder: str = None,
    visualize: bool = True,
    resize_frames: tuple = None,
    downscale_factor: int = 1,
):
    """
    Extract noise from a video by warping random noise using optical flow between consecutive frames.

    If running this function in a Jupyter notebook, you'll see a live preview of the noise and visualization as it calculates.
  
    Args:
        video_path (str): Path to the input video file (MP4), 
                          a folder containing image frames,
                          or a glob pattern like "/path/to/images/*.png".
        noise_channels (int, optional): Number of channels in the generated noise. Defaults to 3.
        output_folder (str, optional): Folder to save the output noise and visualization.
                                       Defaults to None, in which case the folder name is automatically chosen.
        visualize (bool, optional): Whether to generate visualization images and video. Defaults to True.
        resize_frames (tuple or float, optional): Size to resize the input frames.
                                                  If a tuple (height, width), resizes to the exact dimensions.
                                                  If a float, resizes both dimensions relatively and evenly. Defaults to None.
        downscale_factor (int or tuple, optional): Factor(s) by which to downscale the generated noise.
                                                   Larger factor --> smaller noise image.
                                                   This factor should evenly divide the height and width of the video frames.

    Returns:
        tuple: A tuple containing:
            - numpy_noises (np.ndarray): Generated noise with shape [T, C, H, W].
            - vis_frames (np.ndarray): Visualization frames with shape [T, H, W, C].

    Examples:
        # Command line usage
        python noise_warp.py --video_path /path/to/video.mp4 --noise_channels 3 --output_folder /path/to/output
        python noise_warp.py --video_path /path/to/frames_folder --resize_frames 0.5 --downscale_factor 2
        python noise_warp.py --video_path "/path/to/frames/frame_*.png" --resize_frames (256, 256)

        # Function call
        numpy_noises, vis_frames = get_noise_from_video(
            video_path="/path/to/video.mp4",
            noise_channels=3,
            output_folder="/path/to/output",
            visualize=True,
            resize_frames=0.5,
            downscale_factor=(2, 4),
        )
        video_demo("/root/CleanCode/Projects/flow_noise_warping/outputs/water/waves_bilinear.mp4", downscale_factor=4)
        video_demo("/efs/users/ryan.burgert/public/sharing/KevinSpinnerNoiseWarping/diffuse_images_360", downscale_factor=8, resize_frames=.5)
        video_demo("/root/CleanCode/Projects/flow_noise_warping/outputs/kevin_spinner/kevin_vps7.mp4", downscale_factor=4, resize_frames=.5)
    """

    device = rp.select_torch_device(prefer_used=True)
    
    raft_model = raft.RaftOpticalFlow(device, "large")

    # Load video frames into a [T, H, W, C] numpy array, where C=3 and values are between 0 and 1
    # Can be specified as an MP4, a folder that contains images, or a glob like /path/to/*.png
    if rp.is_video_file(video_path):
        video_frames = rp.load_video(video_path)
    else:
        if rp.is_a_folder(video_path):
            frame_paths = rp.get_all_image_files(video_path, sort_by='number')
        else:
            frame_paths = glob.glob(video_path)
            frame_paths = sorted(sorted(frame_paths),key=len)
            if not frame_paths:
                raise ValueError(video_path + " is not a video file, a folder of images, or a glob containing images")
        video_frames = rp.load_images(frame_paths, show_progress=True)

    #If resize_frames is specified, resize all frames to that (height, width)
    if resize_frames is not None:
        rp.fansi_print("Resizing all input frames to size %s"%resize_frames, 'yellow')
        video_frames=rp.resize_images(video_frames, size=resize_frames, interp='area')
        
    video_frames = rp.as_rgb_images(video_frames)
    video_frames = np.stack(video_frames)
    video_frames = video_frames.astype(np.float16)/255
    _, h, w, _ = video_frames.shape
    rp.fansi_print(f"Input video shape: {video_frames.shape}", 'yellow')

    if h%downscale_factor or w%downscale_factor:
        rp.fansi_print("WARNING: height {h} or width{w} is not divisible by the downscale_factor {downscale_factor}. This will lead to artifacts in the noise.")

    # Decide the location of and create the output folder
    if output_folder is None:
        output_folder = "outputs/" + rp.get_file_name(video_path, include_file_extension=False)
    output_folder = rp.make_directory(rp.get_unique_copy_path(output_folder))
    rp.fansi_print("Output folder: " + output_folder, "green")

    with torch.no_grad():

        if visualize and rp.running_in_jupyter_notebook():
            # For previewing results in Jupyter notebooks, if applicable
            display_channel = rp.JupyterDisplayChannel()
            display_channel.display()

        prev_video_frame = video_frames[0]
        noise = torch.randn(noise_channels, h, w).to(device)
        down_noise = rp.torch_resize_image(noise, 1/downscale_factor, interp='area') #Avg pooling
        
        numpy_noise = rp.as_numpy_array(down_noise).astype(np.float16)

        numpy_noises = [numpy_noise]
        vis_frames = []

        try:
            for video_frame in tqdm(video_frames[1:]):

                dx, dy = raft_model(prev_video_frame, video_frame)
                noise = warp_noise(noise, -dx, -dy)
                prev_video_frame = video_frame

                down_noise = rp.torch_resize_image(noise, 1/downscale_factor, interp='area') #Avg pooling
                down_noise = down_noise * downscale_factor #Adjust for STD
                
                numpy_noise = rp.as_numpy_array(down_noise).astype(np.float16)
                numpy_noises.append(numpy_noise)

                if visualize:
                    flow_rgb = rp.optical_flow_to_image(dx, dy)

                    #Turn the noise into a numpy HWC RGB array
                    down_noise_image = np.zeros((*down_noise.shape[1:],3))
                    down_noise_image_c = min(noise_channels,3)
                    down_noise_image[:,:,:down_noise_image_c]=rp.as_numpy_image(down_noise)[:,:,:down_noise_image_c]

                    down_video_frame, down_flow_rgb = rp.resize_images(video_frame, flow_rgb, size=1/downscale_factor, interp='area')
                    
                    visualization = rp.as_byte_image(
                        rp.tiled_images(
                            rp.labeled_images(
                                [
                                    down_noise_image / 3 + 0.5,
                                    down_video_frame,
                                    down_flow_rgb,
                                    down_noise_image / 5 + down_video_frame,
                                ],
                                [
                                    "Warped Noise",
                                    "Input Video",
                                    "Optical Flow",
                                    "Overlaid",
                                ],
                            )
                        )
                    )

                    if rp.running_in_jupyter_notebook():
                        display_channel.update(visualization)

                    vis_frames.append(visualization)

        except KeyboardInterrupt:
            rp.fansi_print("Interrupted! Returning %i noises" % len(numpy_noises), "cyan", "bold")
            pass

    if vis_frames:
        # vis_frames = np.stack(vis_frames)
        vis_img_folder = rp.make_directory(output_folder + "/visualization_images")
        vis_img_paths = rp.path_join(vis_img_folder, "visual_%05i.png")
        rp.save_images(vis_frames, vis_img_paths, show_progress=True)

        if "ffmpeg" in rp.get_system_commands():
            vis_mp4_path = rp.path_join(output_folder, "visualization_video.mp4")
            rp.save_video_mp4(vis_frames, vis_mp4_path, video_bitrate="max", framerate=30)
        else:
            rp.fansi_print("Please install ffmpeg! We won't save an MP4 this time - please try again.")

    numpy_noises = np.stack(numpy_noises).astype(np.float16)
    noises_path = rp.path_join(output_folder, "noises.npy")
    np.save(noises_path, numpy_noises)
    rp.fansi_print("Saved " + noises_path + " with shape " + str(numpy_noises.shape), "green")
    
    rp.fansi_print(rp.get_file_name(__file__)+": Done warping noise, results are at " + rp.get_absolute_path(output_folder), "green", "bold")

    return numpy_noises, vis_frames

if __name__ == '__main__':
    fire.Fire(get_noise_from_video)
