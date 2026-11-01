"""Evaluate matched clean/protected images for multi-face privacy and quality.

Usage is documented in README.md. Detection is performed on the clean image;
each detected region is compared at the same coordinates in its paired image.
"""
import argparse
import csv
import inspect
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from skimage.metrics import structural_similarity
import lpips
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methods import detect_face_boxes
from utils import compute_score, load_model_by_repo_id, pil_to_input

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}


def pairs(clean_dir, protected_dir):
    clean_files = {p.name: p for p in Path(clean_dir).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    protected_files = {p.name: p for p in Path(protected_dir).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    names = sorted(clean_files.keys() & protected_files.keys())
    if not names:
        raise ValueError("No matching image filenames found in clean and protected directories")
    missing = sorted(clean_files.keys() ^ protected_files.keys())
    if missing:
        print(f"Skipping {len(missing)} unmatched filename(s)")
    return [(name, clean_files[name], protected_files[name]) for name in names]


def transform(image, name):
    if name == "resize":
        w, h = image.size
        return image.resize((max(1, int(w * .6)), max(1, int(h * .6))), Image.Resampling.BICUBIC).resize((w, h), Image.Resampling.BICUBIC)
    if name == "crop":
        w, h = image.size
        inset_x, inset_y = int(w * .1), int(h * .1)
        return image.crop((inset_x, inset_y, w - inset_x, h - inset_y)).resize((w, h), Image.Resampling.BICUBIC)
    if name == "jpeg":
        import io
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=50)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if name == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=1.2))
    return image


def image_tensor(image, device):
    return pil_to_input(image).to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean_dir", required=True, help="directory of original images")
    parser.add_argument("--protected_dir", required=True, help="directory of protected images with matching filenames")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--verification_threshold", type=float, required=True,
                        help="cosine threshold calibrated for the selected face verifier and FAR")
    parser.add_argument("--transforms", nargs="+", default=["resize", "crop", "jpeg", "blur"],
                        choices=["resize", "crop", "jpeg", "blur"])
    parser.add_argument("--runtime_csv", default=None,
                        help="optional CSV with filename,runtime_seconds,steps,peak_gpu_memory_mb,face_count")
    parser.add_argument("--clean_edits_dir", default=None,
                        help="optional matched diffusion/rectified-flow edits of clean originals")
    parser.add_argument("--protected_edits_dir", default=None,
                        help="optional matched edits of protected originals (same prompt/seed/settings)")
    parser.add_argument("--edit_model_family", choices=["diffusion", "rectified-flow"], default="diffusion")
    parser.add_argument("--identity_manifest", default=None,
                        help="CSV with filename,face_index,identity_id for valid closed-set re-identification accuracy")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--clip_model_id", default="openai/clip-vit-base-patch32")
    args = parser.parse_args()

    if not -1 <= args.verification_threshold <= 1:
        parser.error("--verification_threshold must be between -1 and 1")
    if bool(args.clean_edits_dir) != bool(args.protected_edits_dir):
        parser.error("provide both edit directories or neither")
    matched_pairs = pairs(args.clean_dir, args.protected_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    if "HF_HOME" not in os.environ:
        parser.error("Set HF_HOME to the Hugging Face cache directory")
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    device = torch.device(args.device)
    fr_id = "minchul/cvlface_adaface_vit_base_kprpe_webface4m"
    aligner_id = "minchul/cvlface_DFA_mobilenet"
    fr_model = load_model_by_repo_id(fr_id, f'{os.environ["HF_HOME"]}/{fr_id}', token).to(device).eval()
    aligner = load_model_by_repo_id(aligner_id, f'{os.environ["HF_HOME"]}/{aligner_id}', token).to(device).eval()
    lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
    clip_model = CLIPModel.from_pretrained(args.clip_model_id).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model_id)

    quality_rows, face_rows, robustness_rows = [], [], []
    embedding_records = []
    for filename, clean_path, protected_path in matched_pairs:
        clean = Image.open(clean_path).convert("RGB")
        protected = Image.open(protected_path).convert("RGB").resize(clean.size)
        a, b = np.asarray(clean), np.asarray(protected)
        mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
        psnr = float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)
        ssim = structural_similarity(a, b, channel_axis=2, data_range=255)
        clean_tensor, protected_tensor = image_tensor(clean, device), image_tensor(protected, device)
        lpips_score = float(lpips_model(clean_tensor, protected_tensor).mean().item())
        clip_inputs = clip_processor(images=[clean, protected], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            clip_features = clip_model.get_image_features(**clip_inputs)
            clip_features = torch.nn.functional.normalize(clip_features, dim=-1)
            clip_similarity = float((clip_features[0] * clip_features[1]).sum().item())
        quality_rows.append({"filename": filename, "psnr_db": psnr, "ssim": ssim,
                             "faces": "", "lpips": lpips_score, "clip_image_similarity": clip_similarity})

        try:
            boxes = detect_face_boxes(image_tensor(clean, device))
        except RuntimeError as exc:
            print(f"Skipping {filename}: {exc}")
            continue
        clean_crops, protected_crops = [], []
        for face_idx, (x1, y1, x2, y2) in enumerate(boxes):
            clean_crops.append(image_tensor(clean.crop((x1, y1, x2, y2)).resize((112, 112)), device))
            protected_crops.append(image_tensor(protected.crop((x1, y1, x2, y2)).resize((112, 112)), device))
        with torch.no_grad():
            scores = compute_score(torch.cat(protected_crops), torch.cat(clean_crops), aligner, fr_model).detach().cpu().tolist()
            aligned, _, landmarks, _, _, _ = aligner(torch.cat(clean_crops))
            signature = inspect.signature(fr_model.model.net.forward)
            embeddings = fr_model(aligned, landmarks) if signature.parameters.get("keypoints") is not None else fr_model(aligned)
            embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1).cpu()
        quality_rows[-1]["faces"] = len(boxes)
        review_dir = Path(args.output_dir) / "boundary_review"
        review_dir.mkdir(parents=True, exist_ok=True)
        review_image = np.asarray(protected).copy()
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(review_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        Image.fromarray(review_image).save(review_dir / filename)
        for face_idx, (box, score) in enumerate(zip(boxes, scores)):
            face_rows.append({"filename": filename, "face_index": face_idx, "cosine_similarity": score,
                              "similarity_drop": 1.0 - score, "verification_attack_success": int(score < args.verification_threshold),
                              "threshold": args.verification_threshold, "box_xyxy": ",".join(map(str, box))})
            embedding_records.append({"filename": filename, "face_index": face_idx,
                                      "clean": embeddings[face_idx], "protected": None})
        # Store protected embeddings for optional identity-labeled re-identification.
        if args.identity_manifest:
            with torch.no_grad():
                aligned_p, _, landmarks_p, _, _, _ = aligner(torch.cat(protected_crops))
                protected_embeddings = fr_model(aligned_p, landmarks_p) if signature.parameters.get("keypoints") is not None else fr_model(aligned_p)
                protected_embeddings = torch.nn.functional.normalize(protected_embeddings.float(), dim=-1).cpu()
            for record, embedding in zip(embedding_records[-len(boxes):], protected_embeddings):
                record["protected"] = embedding

        image_worst = max(scores)
        for transform_name in args.transforms:
            transformed_clean = transform(clean, transform_name)
            transformed_protected = transform(protected, transform_name)
            try:
                trans_boxes = detect_face_boxes(image_tensor(transformed_clean, device))
            except RuntimeError:
                trans_boxes = []
            trans_scores = []
            for x1, y1, x2, y2 in trans_boxes:
                tc = image_tensor(transformed_clean.crop((x1, y1, x2, y2)).resize((112, 112)), device)
                tp = image_tensor(transformed_protected.crop((x1, y1, x2, y2)).resize((112, 112)), device)
                with torch.no_grad():
                    trans_scores.extend(compute_score(tp, tc, aligner, fr_model).detach().cpu().tolist())
            robustness_rows.append({"filename": filename, "transform": transform_name, "faces": len(trans_scores),
                                    "worst_face_similarity": max(trans_scores) if trans_scores else "",
                                    "cosine_similarity_drop_vs_clean": 1.0 - max(trans_scores) if trans_scores else "",
                                    "attack_success_rate": np.mean(np.array(trans_scores) < args.verification_threshold) if trans_scores else ""})
        print(f"{filename}: {len(boxes)} faces; worst cosine similarity {image_worst:.4f}")

    def write_csv(name, rows):
        path = Path(args.output_dir) / name
        columns = sorted({key for row in rows for key in row})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {path}")

    write_csv("image_quality.csv", quality_rows)
    write_csv("per_face_privacy.csv", face_rows)
    write_csv("robustness.csv", robustness_rows)
    if args.identity_manifest:
        labels = {}
        with open(args.identity_manifest, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                labels[(row["filename"], int(row["face_index"]))] = row["identity_id"]
        gallery, queries, truth = [], [], []
        for record in embedding_records:
            key = (record["filename"], record["face_index"])
            if key in labels and record["protected"] is not None:
                gallery.append(record["clean"])
                queries.append(record["protected"])
                truth.append(labels[key])
        if not queries:
            raise ValueError("No detected faces matched rows in --identity_manifest")
        gallery = torch.stack(gallery)
        queries = torch.stack(queries)
        similarities = queries @ gallery.T
        gallery_labels = [labels[(r["filename"], r["face_index"])] for r in embedding_records
                          if (r["filename"], r["face_index"]) in labels]
        predictions = [gallery_labels[i] for i in similarities.argmax(dim=1).tolist()]
        reid_accuracy = float(np.mean(np.array(predictions) == np.array(truth)))
        write_csv("reidentification.csv", [{"top1_accuracy": reid_accuracy, "queries": len(truth),
                                             "gallery_faces": len(gallery_labels), "identity_count": len(set(truth))}])
    if args.clean_edits_dir and args.protected_edits_dir:
        edit_rows = []
        for filename, clean_edit_path, protected_edit_path in pairs(args.clean_edits_dir, args.protected_edits_dir):
            clean_edit = Image.open(clean_edit_path).convert("RGB")
            protected_edit = Image.open(protected_edit_path).convert("RGB").resize(clean_edit.size)
            try:
                edit_boxes = detect_face_boxes(image_tensor(clean_edit, device))
            except RuntimeError:
                continue
            if not edit_boxes:
                continue
            c_crops = [image_tensor(clean_edit.crop(box).resize((112, 112)), device) for box in edit_boxes]
            p_crops = [image_tensor(protected_edit.crop(box).resize((112, 112)), device) for box in edit_boxes]
            with torch.no_grad():
                edit_scores = compute_score(torch.cat(p_crops), torch.cat(c_crops), aligner, fr_model).cpu().tolist()
            edit_rows.append({"filename": filename, "model_family": args.edit_model_family,
                              "faces": len(edit_scores), "worst_face_similarity": max(edit_scores),
                              "attack_success_rate": float(np.mean(np.array(edit_scores) < args.verification_threshold)),
                              "verification_threshold": args.verification_threshold})
        write_csv("editing_robustness.csv", edit_rows)
    if args.runtime_csv:
        with open(args.runtime_csv, newline="", encoding="utf-8") as handle:
            runtime_rows = list(csv.DictReader(handle))
        write_csv("efficiency.csv", runtime_rows)


if __name__ == "__main__":
    main()
