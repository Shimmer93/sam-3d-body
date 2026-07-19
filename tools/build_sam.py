# Copyright (c) Meta Platforms, Inc. and affiliates.

import inspect
import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image


class HumanSegmentor:
    def __init__(self, name="sam2", device="cuda", **kwargs):
        self.device = device

        if name == "sam2":
            print("########### Using human segmentor: SAM2...")
            self.sam = load_sam2(device, **kwargs)
            self.sam_func = run_sam2
        elif name == "sam3":
            print("########### Using human segmentor: SAM3...")
            self.sam = load_sam3(device, **kwargs)
            self.sam_func = run_sam3
        else:
            raise NotImplementedError
    
    def run_sam(self, img, boxes, **kwargs):
        return self.sam_func(self.sam, img, boxes)
        

def load_sam2(device, path):
    checkpoint = f"{path}/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    import sys
    sys.path.append(path)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))
    predictor.model.eval()

    return predictor


def load_sam3(device, path):
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    checkpoint_path = resolve_sam3_checkpoint_path(path)
    model = build_sam3_model(build_sam3_image_model, checkpoint_path, device)
    model = model.to(device)
    model.eval()
    predictor = Sam3Processor(model)
    return predictor


def resolve_sam3_checkpoint_path(path):
    checkpoint_path = path or os.environ.get("SAM3_CHECKPOINT_PATH", "")
    if not checkpoint_path:
        return ""

    checkpoint_path = Path(checkpoint_path).expanduser()
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "sam3.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")
    return str(checkpoint_path)


def build_sam3_model(build_fn, checkpoint_path, device):
    signature = inspect.signature(build_fn)
    kwargs = {}
    if "device" in signature.parameters:
        kwargs["device"] = device

    checkpoint_arg = None
    for name in ("checkpoint_path", "checkpoint", "ckpt_path", "model_path"):
        if name in signature.parameters:
            checkpoint_arg = name
            break
    if checkpoint_path and checkpoint_arg is not None:
        kwargs[checkpoint_arg] = checkpoint_path
        return build_fn(**kwargs)

    model = build_fn(**kwargs)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            for key in ("model", "state_dict", "module"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    checkpoint = value
                    break
        model.load_state_dict(checkpoint, strict=False)
    return model


def run_sam2(sam_predictor, img, boxes):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sam_predictor.set_image(img)
        all_masks, all_scores = [], []
        for i in range(boxes.shape[0]):
            # First prediction: bbox only
            masks, scores, logits = sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes[[i]],
                multimask_output=True,
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]

            mask_1 = masks[0]
            score_1 = scores[0]
            all_masks.append(mask_1)
            all_scores.append(score_1)

            # cv2.imwrite(os.path.join(save_dir, f"{os.path.basename(image_path)[:-4]}_mask_{i}.jpg"), (mask_1 * 255).astype(np.uint8))
        all_masks = np.stack(all_masks)
        all_scores = np.stack(all_scores)

    return all_masks, all_scores


def run_sam3(sam_predictor, img, boxes):
    # SAM3DBodyEstimator passes RGB images to the segmentor path.
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        inference_state = sam_predictor.set_image(img)
        # Prompt the model with text
        output = sam_predictor.set_text_prompt(state=inference_state, prompt="person")

    # Get the masks, bounding boxes, and scores
    masks, _, scores = output["masks"], output["boxes"], output["scores"]
    score_threshold = 0.5
    confident_idx = scores > score_threshold
    masks = masks[confident_idx].float().squeeze(1).cpu().numpy()
    scores = scores[confident_idx].float().cpu().numpy()

    return match_masks_to_boxes(masks, scores, boxes, img.size[::-1])


def match_masks_to_boxes(masks, scores, boxes, image_shape):
    height, width = image_shape
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    matched_masks = np.zeros((len(boxes), height, width), dtype=np.uint8)
    matched_scores = np.zeros(len(boxes), dtype=np.float32)
    if len(masks) == 0 or len(boxes) == 0:
        return matched_masks, matched_scores

    mask_boxes = np.stack([mask_to_box(mask) for mask in masks], axis=0)
    used_masks = set()
    for box_index, box in enumerate(boxes):
        ious = box_iou(box, mask_boxes)
        for mask_index in np.argsort(ious)[::-1]:
            if mask_index not in used_masks:
                break
        else:
            continue
        if ious[mask_index] <= 0:
            continue
        used_masks.add(mask_index)
        matched_masks[box_index] = (masks[mask_index] > 0).astype(np.uint8)
        matched_scores[box_index] = scores[mask_index]
    return matched_masks, matched_scores


def mask_to_box(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.array([0, 0, 0, 0], dtype=np.float32)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def box_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    box_area = np.maximum(box[2] - box[0], 0) * np.maximum(box[3] - box[1], 0)
    boxes_area = (
        np.maximum(boxes[:, 2] - boxes[:, 0], 0)
        * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    )
    return inter / np.maximum(box_area + boxes_area - inter, 1e-6)
