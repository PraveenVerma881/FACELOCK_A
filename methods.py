import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import cv2
import time
from utils import compute_score
import pdb

# CW L2 attack
def cw_l2_attack(X, model, c=0.1, lr=0.01, iters=100, targeted=False):
    encoder = model.vae.encode
    clean_latents = encoder(X).latent_dist.mean

    def f(x):
        latents = encoder(x).latent_dist.mean
        if targeted:
            return latents.norm()
        else:
            return -torch.norm(latents - clean_latents.detach(), p=2, dim=-1)
    
    w = torch.zeros_like(X, requires_grad=True).cuda()
    pbar = tqdm(range(iters))
    optimizer = optim.Adam([w], lr=lr)

    for step in pbar:
        a = 1/2*(nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, X)
        loss2 = torch.sum(c*f(a))

        cost = loss1 + loss2
        pbar.set_description(f"Loss: {cost.item():.5f} | loss1: {loss1.item():.5f} | loss2: {loss2.item():.5f}")
        # pdb.set_trace()

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
    X_adv = 1/2*(nn.Tanh()(w) + 1)
    return X_adv

# Encoder attack - Targeted / Untargeted
def encoder_attack(X, model, eps=0.03, step_size=0.01, iters=100, clamp_min=-1, clamp_max=1, targeted=False):
    """
    Processing encoder attack using l_inf norm
    Params:
        X - image tensor we hope to protect
        model - the targeted edit model
        eps - attack budget
        step_size - attack step size
        iters - attack iterations
        clamp_min - min value for the image pixels
        clamp_max - max value for the image pixels
    Return:
        X_adv - image tensor for the protected image
    """
    encoder = model.vae.encode
    X_adv = torch.clamp(X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).half().cuda(), min=clamp_min, max=clamp_max)
    if not targeted:
        loss_fn = nn.MSELoss()
        clean_latent = encoder(X).latent_dist.mean
    pbar = tqdm(range(iters))
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i

        X_adv.requires_grad_(True)
        latent = encoder(X_adv).latent_dist.mean
        if targeted:
            loss = latent.norm()
            grad, = torch.autograd.grad(loss, [X_adv])
            X_adv = X_adv - grad.detach().sign() * actual_step_size
        else:
            loss = loss_fn(latent, clean_latent)
            grad, = torch.autograd.grad(loss, [X_adv])
            X_adv = X_adv + grad.detach().sign() * actual_step_size

        pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")

        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None

        pbar.set_postfix(norm_2=(X_adv - X).norm().item(), norm_inf=(X_adv - X).abs().max().item())

    return X_adv

def vae_attack(X, model, eps=0.03, step_size=0.01, iters=100, clamp_min=-1, clamp_max=1):
    """
    Processing encoder attack using l_inf norm
    Params:
        X - image tensor we hope to protect
        model - the targeted edit model
        eps - attack budget
        step_size - attack step size
        iters - attack iterations
        clamp_min - min value for the image pixels
        clamp_max - max value for the image pixels
    Return:
        X_adv - image tensor for the protected image
    """
    vae = model.vae
    X_adv = torch.clamp(X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).half().cuda(), min=clamp_min, max=clamp_max)
    pbar = tqdm(range(iters))
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i

        X_adv.requires_grad_()
        image = vae(X_adv).sample

        loss = (image).norm()
        grad, = torch.autograd.grad(loss, [X_adv])
        X_adv = X_adv - grad.detach().sign() * actual_step_size

        pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")

        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None

    return X_adv

def facelock(X, model, aligner, fr_model, lpips_fn, eps=0.03, step_size=0.01, iters=100, clamp_min=-1, clamp_max=1):
    X_adv = torch.clamp(X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).to(X.device), min=clamp_min, max=clamp_max).half()
    pbar = tqdm(range(iters))
    
    vae = model.vae
    X_adv.requires_grad_(True)
    clean_latent = vae.encode(X).latent_dist.mean

    for i in pbar:
        # actual_step_size = step_size
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        
        latent = vae.encode(X_adv).latent_dist.mean
        image = vae.decode(latent).sample.clip(-1, 1)

        loss_cvl = compute_score(image.float(), X.float(), aligner=aligner, fr_model=fr_model)
        loss_encoder = F.mse_loss(latent, clean_latent)
        loss_lpips = lpips_fn(image, X)
        loss = -loss_cvl * (1 if i >= iters * 0.35 else 0.0) + loss_encoder * 0.2 + loss_lpips * (1 if i > iters * 0.25 else 0.0)
        grad, = torch.autograd.grad(loss, [X_adv])
        X_adv = X_adv + grad.detach().sign() * actual_step_size

        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None

        if (i + 1) % 10 == 0 or i + 1 == iters:
            pbar.set_postfix(loss_cvl=loss_cvl.item(), loss_encoder=loss_encoder.item(), loss_lpips=loss_lpips.item(), loss=loss.item())

    return X_adv


def _facelock_batch(X, model, aligner, fr_model, lpips_fn, eps, step_size, iters,
                    clamp_min=-1, clamp_max=1):
    """Warm-start same-sized face crops together with independent per-crop losses."""
    vae = model.vae
    eps_tensor = torch.as_tensor(eps, device=X.device, dtype=X.dtype).view(-1, 1, 1, 1)
    X_adv = torch.clamp(
        X + (torch.rand_like(X) * 2 - 1) * eps_tensor,
        min=clamp_min,
        max=clamp_max,
    ).detach()
    clean_latent = vae.encode(X).latent_dist.mean.detach()

    pbar = tqdm(range(iters))
    for iteration in pbar:
        X_adv = X_adv.detach().requires_grad_(True)
        latent = vae.encode(X_adv).latent_dist.mean
        reconstruction = vae.decode(latent).sample.clamp(-1, 1)
        face_scores = compute_score(
            reconstruction.float(), X.float(), aligner=aligner, fr_model=fr_model
        ).reshape(-1)
        latent_loss = (latent - clean_latent).square().flatten(1).mean(1)
        lpips_loss = lpips_fn(reconstruction, X).reshape(X.shape[0], -1).mean(1)
        loss_cvl_weight = 1.0 if iteration >= iters * 0.35 else 0.0
        loss_lpips_weight = 1.0 if iteration > iters * 0.25 else 0.0
        per_crop_loss = (
            -face_scores * loss_cvl_weight
            + 0.2 * latent_loss
            + loss_lpips_weight * lpips_loss
        )
        gradient, = torch.autograd.grad(per_crop_loss.sum(), [X_adv])
        actual_step_size = step_size - (step_size - step_size / 100) / iters * iteration
        X_adv = X_adv + gradient.detach().sign() * actual_step_size
        X_adv = torch.maximum(torch.minimum(X_adv, X + eps_tensor), X - eps_tensor)
        X_adv = X_adv.clamp(clamp_min, clamp_max).detach()
        if (iteration + 1) % 10 == 0 or iteration + 1 == iters:
            pbar.set_postfix(
                worst_face_similarity=face_scores.max().item(),
                loss=per_crop_loss.mean().item(),
            )
    return X_adv


_FACE_CASCADE = None


def detect_face_boxes(X, min_face_size=24):
    """Detect front-facing faces in a single normalized RGB image tensor."""
    if X.shape[0] != 1:
        raise ValueError("Multi-face FaceLock currently expects one image at a time.")

    image = ((X[0].detach().float().cpu().permute(1, 2, 0).numpy() + 1) * 127.5)
    image = np.clip(image, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    if _FACE_CASCADE.empty():
        raise RuntimeError(f"Could not load OpenCV face detector: {cascade_path}")

    detections = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_face_size, min_face_size),
    )
    height, width = gray.shape
    boxes = []
    for x, y, face_width, face_height in detections:
        pad_x = int(face_width * 0.3)
        pad_y = int(face_height * 0.35)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width, x + face_width + pad_x)
        y2 = min(height, y + face_height + pad_y)
        boxes.append((x1, y1, x2, y2))

    boxes.sort(key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
    if not boxes:
        raise RuntimeError(
            "No faces detected. Use a clear, front-facing image or replace the "
            "OpenCV Haar detector with a stronger detector for profile/occluded faces."
        )
    return boxes


def _face_mask(height, width, device, dtype):
    y = torch.linspace(-1, 1, height, device=device, dtype=dtype)
    x = torch.linspace(-1, 1, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius = (xx / 0.92).square() + (yy / 0.92).square()
    return ((1.0 - radius) / 0.18).clamp(0, 1).view(1, 1, height, width)


def multiface_facelock(
    X,
    model,
    aligner,
    fr_model,
    lpips_fn,
    eps=0.03,
    step_size=0.01,
    iters=100,
    clamp_min=-1,
    clamp_max=1,
    warm_start_iters=None,
    early_stop_similarity=None,
    stats_callback=None,
):
    """Optimize localized face perturbations, then jointly protect the worst face."""
    started = time.perf_counter()
    if X.is_cuda:
        torch.cuda.reset_peak_memory_stats(X.device)
    boxes = detect_face_boxes(X)
    height, width = X.shape[-2:]
    face_regions = [None] * len(boxes)
    weighted_masks = []

    # Warm-start each region independently before composing overlapping masks.
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
    max_area = max(areas)
    if warm_start_iters is None:
        warm_start_iters = max(1, iters // 4)
    crop_buckets = {}
    for face_index, (box, area) in enumerate(zip(boxes, areas)):
        x1, y1, x2, y2 = box
        original_crop = X[:, :, y1:y2, x1:x2]
        crop_height, crop_width = original_crop.shape[-2:]
        padded_height = max(64, ((crop_height + 7) // 8) * 8)
        padded_width = max(64, ((crop_width + 7) // 8) * 8)
        model_crop = F.interpolate(
            original_crop, size=(padded_height, padded_width), mode="bilinear", align_corners=False
        )
        face_weight = max(0.5, (area / max_area) ** 0.25)
        if warm_start_iters > 0:
            crop_buckets.setdefault((padded_height, padded_width), []).append(
                (face_index, model_crop, crop_height, crop_width, face_weight)
            )
        else:
            face_regions[face_index] = torch.zeros_like(original_crop)

    for bucket in crop_buckets.values():
        crop_batch = torch.cat([item[1] for item in bucket], dim=0)
        face_weights = [item[4] for item in bucket]
        if len(bucket) == 1:
            crop_adv_batch = facelock(
                X=crop_batch,
                model=model,
                aligner=aligner,
                fr_model=fr_model,
                lpips_fn=lpips_fn,
                eps=eps * face_weights[0],
                step_size=step_size,
                iters=warm_start_iters,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
            )
        else:
            crop_adv_batch = _facelock_batch(
                X=crop_batch,
                model=model,
                aligner=aligner,
                fr_model=fr_model,
                lpips_fn=lpips_fn,
                eps=[eps * weight for weight in face_weights],
                step_size=step_size,
                iters=warm_start_iters,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
            )

        for crop_index, (face_index, model_crop, crop_height, crop_width, _) in enumerate(bucket):
            x1, y1, x2, y2 = boxes[face_index]
            crop_delta = F.interpolate(
                crop_adv_batch[crop_index:crop_index + 1] - model_crop,
                size=(crop_height, crop_width),
                mode="bilinear",
                align_corners=False,
            ).detach()
            face_regions[face_index] = crop_delta

    for face_index, (box, area) in enumerate(zip(boxes, areas)):
        x1, y1, x2, y2 = box
        crop_height, crop_width = y2 - y1, x2 - x1
        face_weight = max(0.5, (area / max_area) ** 0.25)
        crop_mask = _face_mask(crop_height, crop_width, X.device, X.dtype) * face_weight
        weighted_masks.append((box, crop_mask))

    # Normalize overlapping contributions so the combined perturbation stays bounded.
    numerator = torch.zeros_like(X)
    denominator = torch.zeros_like(X[:, :1])
    for ((x1, y1, x2, y2), crop_mask), crop_delta in zip(weighted_masks, face_regions):
        numerator[:, :, y1:y2, x1:x2] += crop_delta * crop_mask
        denominator[:, :, y1:y2, x1:x2] += crop_mask
    region_mask = denominator.clamp(0, 1)
    warm_delta = numerator / denominator.clamp_min(1e-6)
    delta = (warm_delta * region_mask).clamp(-eps, eps).detach()

    pad_right = (-width) % 8
    pad_bottom = (-height) % 8

    def encode_image(image):
        if pad_right or pad_bottom:
            image = F.pad(image, (0, pad_right, 0, pad_bottom), mode="replicate")
        return model.vae.encode(image).latent_dist.mean

    clean_latent = encode_image(X).detach()
    pbar = tqdm(range(iters))
    executed_iters = 0
    for iteration in pbar:
        delta = delta.detach().requires_grad_(True)
        current = (X + delta).clamp(clamp_min, clamp_max)
        current_latent = encode_image(current)
        reconstruction = model.vae.decode(current_latent).sample[:, :, :height, :width].clamp(-1, 1)
        # Align and recognize all faces in one batch. This avoids two model
        # launches per face at every refinement step, which dominates runtime
        # for group photos with many detections.
        batch_size = 112
        clean_crops, current_crops = [], []
        for x1, y1, x2, y2 in boxes:
            clean_crops.append(F.interpolate(X[:, :, y1:y2, x1:x2].float(),
                                             size=(batch_size, batch_size), mode="bilinear", align_corners=False))
            current_crops.append(F.interpolate(reconstruction[:, :, y1:y2, x1:x2].float(),
                                               size=(batch_size, batch_size), mode="bilinear", align_corners=False))
        clean_batch = torch.cat(clean_crops, dim=0)
        current_batch = torch.cat(current_crops, dim=0)
        identity_scores = compute_score(current_batch, clean_batch, aligner=aligner, fr_model=fr_model)
        if early_stop_similarity is not None and identity_scores.max().item() <= early_stop_similarity:
            break
        loss_identity = identity_scores.max()
        loss_encoder = F.mse_loss(current_latent, clean_latent)
        loss_quality = lpips_fn(reconstruction, X).mean()
        loss_smooth = (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean()
        loss_smooth = loss_smooth + (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()
        mask_edges = F.max_pool2d(region_mask, 3, stride=1, padding=1) - F.avg_pool2d(region_mask, 3, stride=1, padding=1)
        loss_boundary = (delta.abs() * mask_edges).mean()
        loss = loss_identity + 0.2 * loss_encoder + loss_quality + 0.02 * loss_smooth + 0.1 * loss_boundary
        gradient, = torch.autograd.grad(loss, [delta])
        actual_step_size = step_size - (step_size - step_size / 100) / max(iters, 1) * iteration
        delta = (delta - actual_step_size * gradient.sign()).detach()
        delta = delta.clamp(-eps, eps) * region_mask
        executed_iters += 1
        if (iteration + 1) % 5 == 0 or iteration + 1 == iters:
            pbar.set_postfix(
                worst_face_similarity=loss_identity.item(),
                quality=loss_quality.item(),
            )

    result = (X + delta).clamp(clamp_min, clamp_max).detach()
    if stats_callback is not None:
        if X.is_cuda:
            torch.cuda.synchronize(X.device)
            peak_memory_mb = torch.cuda.max_memory_allocated(X.device) / (1024 ** 2)
        else:
            peak_memory_mb = ""
        stats_callback({
            "face_count": len(boxes),
            "warm_start_steps_per_face": int(warm_start_iters),
            "joint_steps": executed_iters,
            "steps_total": len(boxes) * int(warm_start_iters) + executed_iters,
            "runtime_seconds": time.perf_counter() - started,
            "peak_gpu_memory_mb": peak_memory_mb,
        })
    return result
