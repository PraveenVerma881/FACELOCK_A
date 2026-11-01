import inspect
import os
import shutil
import sys

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torchvision.transforms import Compose, Normalize, ToTensor
from transformers import AutoModel


def download(repo_id, path, HF_TOKEN=None):
    """Download the model files declared by a Hugging Face repository."""
    os.makedirs(path, exist_ok=True)
    files_path = os.path.join(path, "files.txt")
    if not os.path.exists(files_path):
        hf_hub_download(repo_id, "files.txt", token=HF_TOKEN, local_dir=path)
    with open(files_path, "r", encoding="utf-8") as handle:
        filenames = handle.read().splitlines()
    for filename in [name for name in filenames if name] + [
        "config.json", "wrapper.py", "model.safetensors"
    ]:
        local_path = os.path.join(path, filename)
        if not os.path.exists(local_path):
            hf_hub_download(repo_id, filename, token=HF_TOKEN, local_dir=path)


def load_model_from_local_path(path, HF_TOKEN=None):
    old_cwd = os.getcwd()
    sys.path.insert(0, path)
    try:
        os.chdir(path)
        return AutoModel.from_pretrained(path, trust_remote_code=True, token=HF_TOKEN)
    finally:
        os.chdir(old_cwd)
        sys.path.pop(0)


def load_model_by_repo_id(repo_id, save_path, HF_TOKEN=None, force_download=False):
    if force_download and os.path.exists(save_path):
        shutil.rmtree(save_path)
    download(repo_id, save_path, HF_TOKEN)
    return load_model_from_local_path(save_path, HF_TOKEN)


def pil_to_input(pil_image):
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform(pil_image).unsqueeze(0)


def compute_score(input1, input2, aligner, fr_model, return_bbox=False):
    aligned1, _, landmarks1, _, _, bbox1 = aligner(input1)
    aligned2, _, landmarks2, _, _, bbox2 = aligner(input2)
    signature = inspect.signature(fr_model.model.net.forward)
    if signature.parameters.get("keypoints") is not None:
        features1 = fr_model(aligned1, landmarks1)
        features2 = fr_model(aligned2, landmarks2)
    else:
        features1 = fr_model(aligned1)
        features2 = fr_model(aligned2)
    similarity = F.cosine_similarity(features1, features2)
    if return_bbox:
        return similarity, bbox1, bbox2
    return similarity


edit_prompts = {
    0: "Turn the person's hair pink",
    1: "Let the person turn bald",
    2: "Let the person have a tatto",
    3: "Let the person wear purple makeup",
    4: "Let the person grow a mustach",
    5: "Turn the person into a zombie",
    6: "Change the skin color to Avatar blue",
    7: "Add elf-like ears",
    8: "Add large vampire fangs",
    9: "Apply Goth style makeup",
    10: "Let the person wear a police suit",
    11: "Let the person wear a bowtie",
    12: "Let the person wear a helmet",
    13: "Let the person wear sunglasses",
    14: "Let the person wear earrings",
    15: "Let the person smoke a cigar",
    16: "Place a headband in the hair",
    17: "Place a tiara on the top of the head",
    18: "Let it be snowy",
    19: "Change the background to a beach",
    20: "Add a city skyline background",
    21: "Add a forest background",
    22: "Change the background to a desert",
    23: "Set the background in a library",
    24: "Let the person stand under the moon",
}


def get_image_embeddings(images, clip_processor, clip_model, device="cuda"):
    inputs = clip_processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        return clip_model.get_image_features(**inputs)


def get_text_embeddings(prompts, clip_processor, clip_model, device="cuda"):
    inputs = clip_processor(
        text=prompts, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    with torch.no_grad():
        return clip_model.get_text_features(**inputs)
