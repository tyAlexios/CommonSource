"""
Sa2VA: State-of-the-art vision-language model for image/video understanding.

This module provides simple functions to work with ByteDance's Sa2VA-4B model, 
which handles both images and videos for:
- Image and video captioning
- Visual question answering 
- Referring segmentation (generates pixel masks for objects)

Functions automatically download the model on first use. No class initialization needed.

Input formats:
- Images: np.ndarray (HW3 uint8 or float 0-1), PIL Image, or path/URL
- Videos: List of frames, path, or URL
- Text: String prompts for questions or segmentation references

Example:
    # Caption an image
    caption = describe_image("photo.jpg")
    
    # Ask questions about an image
    answer = chat_image("photo.jpg", "What color is the car?")
    
    # Generate segmentation mask for a referred object
    text, masks = segment_image("photo.jpg", "Please segment the cat")

See: https://huggingface.co/ByteDance/Sa2VA-4B
"""
import rp
import torch
from PIL import Image


__all__ = [
    "describe_image", 
    "describe_video",
    "chat_image", 
    "chat_video",
    "segment_image",
    "segment_video"
]


_sa2va_device = None


def _default_sa2va_device():
    """
    Get or initialize the default device for Sa2VA model.
    Uses rp.select_torch_device() to pick the best available device.
    This global device is updated whenever a model is loaded with 
    a specific device.
    
    Returns:
        The default torch device for Sa2VA
    """
    global _sa2va_device
    if _sa2va_device is None:
        _sa2va_device = rp.select_torch_device(reserve=True)
    return _sa2va_device


@rp.memoized
def _get_sa2va_model_helper(path, device):
    """
    Helper function to download and set up the Sa2VA model.
    Results are memoized for efficiency. Sets the global _sa2va_device.
    
    Args:
        path: HuggingFace model path
        device: Device to load the model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # These packages are needed
    rp.pip_import("timm")
    rp.pip_import("flash_attn")
    rp.pip_import("transformers")

    from transformers import AutoTokenizer, AutoModel

    if device is None:
        device = _default_sa2va_device()
    else:
        global _sa2va_device
        _sa2va_device = device

    # load the model and tokenizer
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval()

    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=False
    )

    return model, tokenizer


def _get_sa2va_model(path="ByteDance/Sa2VA-4B", device=None):
    """
    Get the Sa2VA model and tokenizer. Downloads from HuggingFace if not cached.
    The model will be loaded onto the specified device, or a default device 
    if none is specified. The device used becomes the new default device.
    
    Args:
        path: HuggingFace model path
        device: Device to load the model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    return _get_sa2va_model_helper(path, device)


def _load_video(video, num_frames=8):
    """
    Load and preprocess a video for Sa2VA model input.
    
    Args:
        video: A video as path, URL, or list of frames
        num_frames: Number of frames to sample from the video
        
    Returns:
        List of PIL.Image objects ready for the model
    """
    # Load the video if its a path or URL
    if isinstance(video, str):
        video = rp.load_video(video)

    # Evenly space the frames
    video = rp.resize_list(video, num_frames)

    # Convert to PIL images
    video = rp.as_numpy_images(video)
    video = rp.as_rgb_images(video)
    video = rp.as_byte_images(video)
    video = rp.as_pil_images(video)

    return video


def _load_image(image):
    """
    Load and preprocess an image for Sa2VA model input.
    
    Args:
        image: An image as path, URL, np.ndarray, or PIL.Image
        
    Returns:
        PIL.Image object ready for the model
    """
    # Load the image if its a path or URL
    if isinstance(image, str):
        image = rp.load_image(image)

    # Convert to PIL image
    image = rp.as_numpy_image(image)
    image = rp.as_rgb_image(image)
    image = rp.as_byte_image(image)
    image = rp.as_pil_image(image)

    return image


def _run_sa2va(content, prompt, is_video=False, device=None, return_masks=False) -> str | tuple[str, list]:
    """
    Internal helper function that handles both image and video inputs.
    
    Args:
        content: Image or video input
        prompt: Text prompt for querying the model
        is_video: Whether the content is a video
        device: Device to run the model on
        return_masks: Whether to return segmentation masks
        
    Returns:
        String response or tuple of (response, masks) if return_masks=True
    """
    model, tokenizer = _get_sa2va_model(device=device)
    
    # Load and process the content
    if is_video:
        content = _load_video(content)
        content_key = "video"
    else:
        content = _load_image(content)
        content_key = "image"
    
    # Prepare the prompt
    text_prompts = "<image>" + prompt
    
    # Build input dictionary
    input_dict = {
        content_key: content,
        "text": text_prompts,
        "past_text": "",
        "mask_prompts": None,
        "tokenizer": tokenizer,
    }
    
    # Run the model
    with torch.no_grad():
        return_dict = model.predict_forward(**input_dict)
    
    # Process the output
    predicted_text = return_dict["prediction"]
    predicted_text = predicted_text.rstrip("<|end|>")
    
    if return_masks and 'prediction_masks' in return_dict:
        return predicted_text, return_dict['prediction_masks']
    
    return predicted_text


def chat_image(image, prompt, device=None) -> str:
    """
    Given an image and a text prompt, return text.
    Can answer questions about the image, etc.
    
    Args:
        image: np.ndarray, PIL Image, or path/URL
        prompt: Text prompt for querying the model
        device: Optional device to run inference on. If any Sa2VA model has been 
               initialized previously, the most recent device becomes the default.
        
    Returns:
        Text response from the model
    """
    return _run_sa2va(image, prompt, is_video=False, device=device)


def chat_video(video, prompt, device=None) -> str:
    """
    Given a video and a text prompt, return text.
    Can caption videos or answer questions about it, etc.
    
    Args:
        video: List of frames, path, or URL
        prompt: Text prompt for querying the model
        device: Optional device to run inference on. If any Sa2VA model has been 
               initialized previously, the most recent device becomes the default.
        
    Returns:
        Text response from the model
    """
    return _run_sa2va(video, prompt, is_video=True, device=device)


def describe_image(image, device=None) -> str:
    """
    Image captioning: generates a description of the given image.
    
    Args:
        image: np.ndarray, PIL Image, or path/URL
        device: Optional device to run inference on. If any Sa2VA model has been 
               initialized previously, the most recent device becomes the default.
        
    Returns:
        Text description of the image
    """
    return _run_sa2va(image, "Generate a detailed description of the image.", is_video=False, device=device)


def describe_video(video, device=None) -> str:
    """
    Video captioning: generates a description of the given video.
    
    Args:
        video: List of frames, path, or URL 
        device: Optional device to run inference on. If any Sa2VA model has been 
               initialized previously, the most recent device becomes the default.
        
    Returns:
        Text description of the video
    """
    return _run_sa2va(video, "Generate a detailed description of the video.", is_video=True, device=device)


def segment_image(image, prompt, device=None) -> tuple[str, list]:
    """
    Performs referring segmentation on an image based on the text prompt.
    
    Args:
        image: np.ndarray, PIL Image, or path/URL
        prompt: Text prompt describing what to segment (e.g., "Please segment the person")
        device: Optional device to run inference on. If any Sa2VA model has been 
               initialized previously, the most recent device becomes the default.
        
    Returns:
        Tuple of (text_response, segmentation_masks)
        - text_response: The model's text response
        - segmentation_masks: Binary masks for the referred objects
    """
    return _run_sa2va(image, prompt, is_video=False, device=device, return_masks=True)


def segment_video(video, prompt, device=None) -> tuple[str, list]:
    """
    Performs referring segmentation on a video based on the text prompt.
    
    Args:
        video: List of frames, path, or URL
        prompt: Text prompt describing what to segment (e.g., "Please segment the person")
        device: Optional device to run inference on. If any Sa2VA model has been 
               initialized previously, the most recent device becomes the default.
        
    Returns:
        Tuple of (text_response, segmentation_masks)
        - text_response: The model's text response
        - segmentation_masks: List of binary masks for each frame
    """
    return _run_sa2va(video, prompt, is_video=True, device=device, return_masks=True)


