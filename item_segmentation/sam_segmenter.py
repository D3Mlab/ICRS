from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import os

class SamSegmenter:
    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint_path: Optional[str] = None,
        device_str: str = "cpu",
    ) -> None:
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = self._resolve_device(device_str)
        self._sam = None

    def _resolve_device(self, device_str: str) -> torch.device:
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if device_str == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        if device_str == "mps" and not torch.backends.mps.is_available():
            return torch.device("cpu")
        return torch.device(device_str)

    def _download_default_checkpoint(self) -> str:
        urls = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        }
        if self.model_type not in urls:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        cache_dir = Path.home() / ".cache" / "segment_anything"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = cache_dir / f"sam_{self.model_type}.pth"
        if ckpt_path.exists():
            return str(ckpt_path)
        import urllib.request
        urllib.request.urlretrieve(urls[self.model_type], ckpt_path)
        return str(ckpt_path)

    def load(self) -> None:
        from segment_anything import sam_model_registry

        ckpt = self.checkpoint_path or self._download_default_checkpoint()
        sam = sam_model_registry[self.model_type](checkpoint=ckpt)
        sam.to(device=self.device)
        self._sam = sam

    @staticmethod
    def maybe_resize_image(image_bgr: np.ndarray, max_side: int) -> Tuple[np.ndarray, float, float]:
        if max_side is None or max_side <= 0:
            h, w = image_bgr.shape[:2]
            return image_bgr, 1.0, 1.0
        h, w = image_bgr.shape[:2]
        longest = max(h, w)
        if longest <= max_side:
            return image_bgr, 1.0, 1.0
        scale = max_side / float(longest)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        sx = w / float(new_w)
        sy = h / float(new_h)
        return resized, sx, sy

    @staticmethod
    def _bounding_box_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
        ys, xs = np.where(mask)
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    @staticmethod
    def _expand_box(x_min: int, y_min: int, x_max: int, y_max: int, pad: int, w: int, h: int) -> Tuple[int, int, int, int]:
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w - 1, x_max + pad)
        y_max = min(h - 1, y_max + pad)
        return x_min, y_min, x_max, y_max

    def generate_masks(
        self,
        image_bgr: np.ndarray,
        points_per_side: int = 16,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_n_points_downscale_factor: int = 2,
        min_mask_region_area: int = 64,
    ) -> List[Dict]:
        if self._sam is None:
            self.load()
        from segment_anything import SamAutomaticMaskGenerator

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        generator = SamAutomaticMaskGenerator(
            model=self._sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            box_nms_thresh=box_nms_thresh,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
        )
        return generator.generate(image_rgb)

    def resize_masks_to_image(self, masks: List[Dict], target_w: int, target_h: int) -> List[Dict]:
        resized_masks: List[Dict] = []
        for mask_dict in masks:
            seg = mask_dict.get("segmentation")
            if seg is None:
                continue
            seg_uint8 = seg.astype(np.uint8)
            seg_back = cv2.resize(seg_uint8, (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            updated = dict(mask_dict)
            updated["segmentation"] = seg_back
            updated["area"] = int(seg_back.sum())
            resized_masks.append(updated)
        return resized_masks

    def save_crops(
        self,
        image_bgr: np.ndarray,
        masks: List[Dict],
        output_dir: Path,
        min_area: int = 500,
        padding: int = 4,
    ) -> int:
        h, w = image_bgr.shape[:2]
        saved_count = 0
        output_dir.mkdir(parents=True, exist_ok=True)
        for idx, mask_dict in enumerate(masks):
            mask = mask_dict.get("segmentation")
            if mask is None:
                continue
            area = int(mask_dict.get("area", int(mask.sum())))
            if area < min_area:
                continue
            x_min, y_min, x_max, y_max = self._bounding_box_from_mask(mask)
            x_min, y_min, x_max, y_max = self._expand_box(x_min, y_min, x_max, y_max, padding, w, h)
            crop = image_bgr[y_min : y_max + 1, x_min : x_max + 1]
            crop_mask = mask[y_min : y_max + 1, x_min : x_max + 1]
            if crop.size == 0 or crop_mask.size == 0:
                continue
            base = output_dir / f"segment_{idx:04d}"
            crop_path = os.path.join(output_dir, 'crops')
            os.makedirs(crop_path, exist_ok=True)
            cv2.imwrite(os.path.join(crop_path, f"segment_{idx:04d}.jpg"), crop)
            rgba = np.dstack([
                cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                (crop_mask.astype(np.uint8) * 255),
            ])
            rgba_path = os.path.join(output_dir, 'objects')
            os.makedirs(rgba_path, exist_ok=True)
            rgba_path = os.path.join(rgba_path, f"segment_{idx:04d}.png")
            cv2.imwrite(rgba_path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
            saved_count += 2
        return saved_count

    def segment_image(
        self,
        image_path: Path,
        output_dir: Path,
        min_area: int = 500,
        padding: int = 4,
        max_side: int = 1024,
        points_per_side: int = 16,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_n_points_downscale_factor: int = 2,
        min_mask_region_area: int = 64,
        limit_masks: int = 0,
        exhaustive: bool = False,
    ) -> Tuple[int, List[Dict]]:
        image_bgr_orig = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr_orig is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        # Exhaustive preset for higher recall of segments
        if exhaustive:
            points_per_side = max(points_per_side, 64)
            crop_n_layers = max(crop_n_layers, 2)
            crop_n_points_downscale_factor = min(crop_n_points_downscale_factor, 1)
            pred_iou_thresh = min(pred_iou_thresh, 0.80)
            stability_score_thresh = min(stability_score_thresh, 0.90)
            box_nms_thresh = max(box_nms_thresh, 0.90)
            min_mask_region_area = min(min_mask_region_area, 16)
            # keep all masks by default in exhaustive mode
            limit_masks = 0
            # increase resolution cap if reasonably small
            if max_side and max_side > 0 and max_side < 1536:
                max_side = 1536

        image_bgr_proc, _, _ = self.maybe_resize_image(image_bgr_orig, max_side)

        try:
            masks = self.generate_masks(
                image_bgr=image_bgr_proc,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                box_nms_thresh=box_nms_thresh,
                crop_n_layers=crop_n_layers,
                crop_n_points_downscale_factor=crop_n_points_downscale_factor,
                min_mask_region_area=min_mask_region_area,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise RuntimeError("Out of memory during mask generation. Reduce resolution or sampler density.")
            raise

        if limit_masks and limit_masks > 0:
            masks = sorted(masks, key=lambda m: int(m.get("area", 0)), reverse=True)[:limit_masks]

        h0, w0 = image_bgr_orig.shape[:2]
        masks_resized = self.resize_masks_to_image(masks, w0, h0)

        saved = self.save_crops(image_bgr_orig, masks_resized, output_dir, min_area=min_area, padding=padding)
        return saved, masks_resized


