import rp
import torch


@rp.memoized
def _default_sa2va_device():
    return rp.select_torch_device()


@rp.memoized
def _get_sa2va_model_helper(path, device):

    # These packages are needed
    rp.pip_import("timm")
    rp.pip_import("flash_attn")
    rp.pip_import("transformers")

    from transformers import AutoTokenizer, AutoModel

    if device is None:
        device = _default_sa2va_device()

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


def _load_video(video, num_frames):
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


def run_video_chat(video, prompt, device=None) -> str:
    """
    Given a video and a text prompt, return text.
    Can caption videos or answer questions about it, etc.
    """

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

    predicted_text = return_dict["prediction"]  # the text format answer
    predicted_text = predicted_text.rstip("<|end|>")

    return preficted_text
