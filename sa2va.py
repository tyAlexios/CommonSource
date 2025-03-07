import rp
import torch
from PIL import Image


_sa2va_device = None


def _default_sa2va_device():
    global _sa2va_device
    if _sa2va_device is None:
        _sa2va_device = rp.select_torch_device()
    return _sa2va_device


@rp.memoized
def _get_sa2va_model_helper(path, device):

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

    return rp.gather_vars("model tokenizer")


def _get_sa2va_model(path="ByteDance/Sa2VA-4B", device=None):
    """From https://huggingface.co/ByteDance/Sa2VA-4B"""
    return _get_sa2va_model_helper(path, device)


def _load_video(video, num_frames=8):
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
    # Load the image if its a path or URL
    if isinstance(image, str):
        image = rp.load_image(image)

    # Convert to PIL image
    image = rp.as_numpy_image(image)
    image = rp.as_rgb_image(image)
    image = rp.as_byte_image(image)
    image = rp.as_pil_image(image)

    return image


def describe_image(image, device=None) -> str:
    """
    Image captioning: generates a description of the given image.
    """
    model, tokenizer = _get_sa2va_model(device=device)

    image = _load_image(image)

    text_prompts = "<image>Generate a detailed description of the image."

    input_dict = {
        "image": image,
        "text": text_prompts,
        "past_text": "",
        "mask_prompts": None,
        "tokenizer": tokenizer,
    }

    with torch.no_grad():
        return_dict = model.predict_forward(**input_dict)

    predicted_text = return_dict["prediction"]
    predicted_text = predicted_text.rstrip("<|end|>")

    return predicted_text


def describe_video(video, device=None) -> str:
    """
    Video captioning: generates a description of the given video.
    """
    model, tokenizer = _get_sa2va_model(device=device)

    video = _load_video(video)

    text_prompts = "<image>Generate a detailed description of the video."

    input_dict = {
        "video": video,
        "text": text_prompts,
        "past_text": "",
        "mask_prompts": None,
        "tokenizer": tokenizer,
    }

    with torch.no_grad():
        return_dict = model.predict_forward(**input_dict)

    predicted_text = return_dict["prediction"]
    predicted_text = predicted_text.rstrip("<|end|>")

    return predicted_text


def run_video_chat(video, prompt, device=None) -> str:
    """
    Given a video and a text prompt, return text.
    Can caption videos or answer questions about it, etc.
    """
    model, tokenizer = _get_sa2va_model(device=device)

    video = _load_video(video)

    text_prompts = "<image>" + prompt

    input_dict = {
        "video": video,
        "text": text_prompts,
        "past_text": "",
        "mask_prompts": None,
        "tokenizer": tokenizer,
    }

    with torch.no_grad():
        return_dict = model.predict_forward(**input_dict)

    predicted_text = return_dict["prediction"]
    predicted_text = predicted_text.rstrip("<|end|>")

    return predicted_text


def run_image_chat(image, prompt, device=None) -> str:
    """
    Given an image and a text prompt, return text.
    Can answer questions about the image, etc.
    """
    model, tokenizer = _get_sa2va_model(device=device)

    image = _load_image(image)

    text_prompts = "<image>" + prompt

    input_dict = {
        "image": image,
        "text": text_prompts,
        "past_text": "",
        "mask_prompts": None,
        "tokenizer": tokenizer,
    }

    with torch.no_grad():
        return_dict = model.predict_forward(**input_dict)

    predicted_text = return_dict["prediction"]
    predicted_text = predicted_text.rstrip("<|end|>")

    return predicted_text
