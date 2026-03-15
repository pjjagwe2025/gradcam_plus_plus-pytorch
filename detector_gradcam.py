import os
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw


def resolve_module(model, layer_name: str):
    """Resolve a dotted module path like 'model.22.cv3.1.1'."""
    if hasattr(model, "get_submodule"):
        return model.get_submodule(layer_name)

    module = model
    for part in layer_name.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def list_named_modules(model, n: int = 40):
    """Print the last n named modules to help choose a target layer."""
    items = list(model.named_modules())
    for name, module in items[-n:]:
        print(f"{name:40s} -> {module.__class__.__name__}")


def find_last_conv_layer(model, exclude_keywords=("dfl",)):
    """Return the name and module of the last Conv2d layer."""
    items = list(model.named_modules())
    for name, module in reversed(items):
        if isinstance(module, nn.Conv2d):
            lname = name.lower()
            if not any(k in lname for k in exclude_keywords):
                return name, module
    raise ValueError("No Conv2d layer found.")


def letterbox_image(image_rgb: np.ndarray, new_shape=(1280, 1280), color=(114, 114, 114)):
    """
    Resize + pad like YOLO letterbox.
    Returns: letterboxed_image, ratio, (dw, dh)
    """
    shape = image_rgb.shape[:2]  # (h, w)

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w, h)
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image_rgb = cv2.resize(image_rgb, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    image_lb = cv2.copyMakeBorder(
        image_rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return image_lb, r, (dw, dh)


def prepare_input_tensor(image_path: str, img_size=1280, device=None):
    """
    Read image, apply letterbox, convert to float tensor in [0, 1].
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image_lb, ratio, pad = letterbox_image(image_rgb, (img_size, img_size))
    tensor = torch.from_numpy(image_lb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    if device is not None:
        tensor = tensor.to(device)

    meta = {
        "orig_shape": image_rgb.shape[:2],   # (h, w)
        "letterbox_shape": image_lb.shape[:2],
        "ratio": ratio,
        "pad": pad,  # (dw, dh)
        "image_rgb": image_rgb,
        "image_lb": image_lb,
        "image_path": image_path,
    }
    return tensor, meta


def cam_to_original(cam: np.ndarray, meta: dict):
    """
    Map CAM from letterboxed image space back to original image size.
    """
    if cam.ndim == 3:
        cam = cam.squeeze()

    h0, w0 = meta["orig_shape"]
    h1, w1 = meta["letterbox_shape"]
    dw, dh = meta["pad"]

    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))

    y1 = top
    y2 = h1 - bottom
    x1 = left
    x2 = w1 - right

    cam = cam[y1:y2, x1:x2]
    cam = cv2.resize(cam, (w0, h0), interpolation=cv2.INTER_LINEAR)

    cam = cam - cam.min()
    denom = cam.max() - cam.min()
    if denom > 0:
        cam = cam / denom
    return cam


def overlay_cam_on_image(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.35):
    """
    Overlay a normalized CAM (0..1) on an RGB image.
    """
    if cam.ndim == 3:
        cam = cam.squeeze()

    cam_uint8 = np.uint8(np.clip(cam, 0, 1) * 255)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap_rgb + (1 - alpha) * image_rgb).astype(np.uint8)
    return heatmap_rgb, overlay


def _color_for_class(class_id: int):
    palette = [
        (255, 99, 71), (60, 179, 113), (65, 105, 225), (255, 165, 0),
        (138, 43, 226), (220, 20, 60), (46, 139, 87), (30, 144, 255),
    ]
    return palette[class_id % len(palette)]


def draw_boxes(
    image_rgb: np.ndarray,
    boxes,
    labels,
    scores=None,
    class_names=None,
    box_format="xyxy",
    width=3,
):
    """
    Draw boxes on an RGB image. boxes expected as iterable of [x1,y1,x2,y2].
    """
    img = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(img)

    for i, box in enumerate(boxes):
        if box_format != "xyxy":
            raise ValueError("Only xyxy format is supported in draw_boxes.")

        x1, y1, x2, y2 = [float(v) for v in box]
        class_id = int(labels[i])
        color = _color_for_class(class_id)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        cls_name = class_names[class_id] if class_names is not None and class_id < len(class_names) else str(class_id)
        score_txt = f" {scores[i]:.3f}" if scores is not None else ""
        text = f"{cls_name}{score_txt}"

        tx = max(0, x1)
        ty = max(0, y1 - 16)
        draw.text((tx, ty), text, fill=color)

    return np.array(img)


def read_yolo_txt_labels(label_file: str, image_width: int, image_height: int):
    """
    Read YOLO txt labels and convert to xyxy.
    Returns boxes, class_ids.
    """
    boxes = []
    class_ids = []

    if not os.path.exists(label_file):
        return boxes, class_ids

    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])

            x1 = (xc - bw / 2) * image_width
            y1 = (yc - bh / 2) * image_height
            x2 = (xc + bw / 2) * image_width
            y2 = (yc + bh / 2) * image_height

            boxes.append([x1, y1, x2, y2])
            class_ids.append(cls_id)

    return boxes, class_ids


class ActivationsAndGradients:
    def __init__(self, target_layer):
        self.activations = None
        self.gradients = None
        self.handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def release(self):
        for h in self.handles:
            h.remove()


class DetectorCAMBase:
    """
    Detector-friendly Grad-CAM / Grad-CAM++.

    Supported detector types:
      - 'ultralytics' : YOLO / RT-DETR loaded through ultralytics
      - 'torchvision' : FasterRCNN / RetinaNet / torchvision detectors

    Targeting modes:
      - class_idx=None: explain the single highest-scoring detection overall
      - class_idx=<int>, detection_index=None: explain top-k detections of a given class
      - class_idx=<int>, detection_index=<int>: explain one chosen detection
    """

    def __init__(self, model, target_layer, detector_type="ultralytics", class_names=None, device=None):
        self.model = model.eval()
        self.detector_type = detector_type.lower()
        self.device = device or next(model.parameters()).device

        if isinstance(target_layer, str):
            target_layer = resolve_module(model, target_layer)
        self.target_layer = target_layer

        self.hooks = ActivationsAndGradients(self.target_layer)
        self.class_names = self._normalize_class_names(class_names)

    @staticmethod
    def _normalize_class_names(class_names):
        if class_names is None:
            return None
        if isinstance(class_names, dict):
            return [class_names[k] for k in sorted(class_names.keys())]
        return list(class_names)

    def _get_num_classes(self):
        if self.class_names is not None:
            return len(self.class_names)

        for attr_name in ("nc", "num_classes"):
            if hasattr(self.model, attr_name):
                value = getattr(self.model, attr_name)
                if isinstance(value, int):
                    return value

        if hasattr(self.model, "model") and len(self.model.model) > 0:
            head = self.model.model[-1]
            for attr_name in ("nc", "num_classes"):
                if hasattr(head, attr_name):
                    value = getattr(head, attr_name)
                    if isinstance(value, int):
                        return value
        return None

    def _collect_tensors(self, obj):
        tensors = []
        if torch.is_tensor(obj):
            tensors.append(obj)
        elif isinstance(obj, (list, tuple)):
            for x in obj:
                tensors.extend(self._collect_tensors(x))
        elif isinstance(obj, dict):
            for x in obj.values():
                tensors.extend(self._collect_tensors(x))
        return tensors

    def _choose_ultralytics_tensor(self, outputs):
        nc = self._get_num_classes()
        candidates = [t for t in self._collect_tensors(outputs) if torch.is_tensor(t) and t.ndim == 3]
        if not candidates:
            raise ValueError("Could not find a 3D detection tensor inside the model outputs.")

        def orientation_score(shape):
            n, c = shape
            score = 0
            if c >= 6:
                score += 2
            if n >= c:
                score += 1
            if c <= 1024:
                score += 1
            if nc is not None:
                if c in {4 + nc, 5 + nc}:
                    score += 5
                elif c >= 4 + nc and c <= 4 + nc + 128:
                    score += 3
            return score

        best = None
        best_score = -1
        best_perm = False

        for t in candidates:
            keep_score = orientation_score((t.shape[1], t.shape[2]))  # [B, N, C]
            perm_score = orientation_score((t.shape[2], t.shape[1]))  # [B, N, C] after permute

            if keep_score > best_score:
                best = t
                best_score = keep_score
                best_perm = False
            if perm_score > best_score:
                best = t
                best_score = perm_score
                best_perm = True

        if best_perm:
            best = best.permute(0, 2, 1).contiguous()

        return best  # [B, N, C]

    def _split_ultralytics_scores(self, det_tensor):
        """
        det_tensor: [B, N, C]
        Returns combined_scores [B, N, nc]
        """
        nc = self._get_num_classes()
        c = det_tensor.shape[-1]

        obj = None
        if nc is not None:
            cls = det_tensor[..., -nc:]
            if c == 5 + nc:
                obj = det_tensor[..., 4:5]
        else:
            # fallback heuristic
            cls = det_tensor[..., 4:]
            if c >= 6:
                obj = det_tensor[..., 4:5]

        combined = cls
        if obj is not None and obj.shape[-1] == 1:
            combined = cls * obj

        return combined, cls, obj

    def _ultralytics_target(self, outputs, class_idx=None, topk=10, detection_index=None):
        det = self._choose_ultralytics_tensor(outputs)  # [B, N, C]
        combined_scores, cls_scores, obj_scores = self._split_ultralytics_scores(det)

        scores = combined_scores[0]  # [N, nc]
        if scores.ndim != 2:
            raise ValueError(f"Expected a 2D [N, nc] score tensor, got shape={tuple(scores.shape)}")

        if class_idx is None:
            flat_index = scores.argmax()
            det_idx = int(flat_index // scores.shape[1])
            cls_idx = int(flat_index % scores.shape[1])
            scalar = scores[det_idx, cls_idx]
            info = {
                "mode": "top_detection",
                "detection_index": det_idx,
                "class_idx": cls_idx,
                "score": float(scalar.detach().cpu()),
            }
            return scalar, info

        if detection_index is not None:
            scalar = scores[detection_index, class_idx]
            info = {
                "mode": "single_detection",
                "detection_index": int(detection_index),
                "class_idx": int(class_idx),
                "score": float(scalar.detach().cpu()),
            }
            return scalar, info

        class_scores = scores[:, class_idx]
        k = min(int(topk), class_scores.numel())
        vals, idx = torch.topk(class_scores, k=k)
        scalar = vals.sum()

        info = {
            "mode": "class_topk_sum",
            "class_idx": int(class_idx),
            "topk": int(k),
            "top_indices": idx.detach().cpu().tolist(),
            "top_scores": vals.detach().cpu().tolist(),
            "score_sum": float(scalar.detach().cpu()),
        }
        return scalar, info

    def _torchvision_target(self, outputs, class_idx=None, topk=10, detection_index=None):
        if not isinstance(outputs, (list, tuple)) or len(outputs) == 0:
            raise ValueError("Torchvision detector output must be a non-empty list/tuple of dicts.")

        out = outputs[0]
        if not isinstance(out, dict) or "scores" not in out or "labels" not in out:
            raise ValueError("Torchvision detector output dict must contain 'scores' and 'labels'.")

        scores = out["scores"]
        labels = out["labels"]

        if scores.numel() == 0:
            raise ValueError("No predictions returned by torchvision detector.")

        if class_idx is None:
            det_idx = int(scores.argmax())
            scalar = scores[det_idx]
            cls_idx = int(labels[det_idx])
            info = {
                "mode": "top_detection",
                "detection_index": det_idx,
                "class_idx": cls_idx,
                "score": float(scalar.detach().cpu()),
            }
            return scalar, info

        mask = labels == class_idx
        idxs = torch.where(mask)[0]
        if idxs.numel() == 0:
            raise ValueError(f"No detections found for class_idx={class_idx}.")

        if detection_index is not None:
            scalar = scores[detection_index]
            info = {
                "mode": "single_detection",
                "detection_index": int(detection_index),
                "class_idx": int(class_idx),
                "score": float(scalar.detach().cpu()),
            }
            return scalar, info

        class_scores = scores[idxs]
        k = min(int(topk), class_scores.numel())
        vals, order = torch.topk(class_scores, k=k)
        chosen = idxs[order]
        scalar = vals.sum()

        info = {
            "mode": "class_topk_sum",
            "class_idx": int(class_idx),
            "topk": int(k),
            "top_indices": chosen.detach().cpu().tolist(),
            "top_scores": vals.detach().cpu().tolist(),
            "score_sum": float(scalar.detach().cpu()),
        }
        return scalar, info

    def _build_target(self, outputs, class_idx=None, topk=10, detection_index=None):
        if self.detector_type == "ultralytics":
            return self._ultralytics_target(outputs, class_idx, topk, detection_index)
        if self.detector_type == "torchvision":
            return self._torchvision_target(outputs, class_idx, topk, detection_index)
        raise ValueError(f"Unsupported detector_type: {self.detector_type}")

    @staticmethod
    def _normalize_cam(cam):
        cam = cam.detach()
        cam = cam - cam.amin(dim=(2, 3), keepdim=True)
        cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)
        return cam

    def _forward_and_backward(self, input_tensor, scalar_score):
        self.model.zero_grad(set_to_none=True)
        scalar_score.backward(retain_graph=False)
        activations = self.hooks.activations
        gradients = self.hooks.gradients

        if activations is None or gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients. Check target_layer.")
        return activations, gradients

    def __call__(self, input_tensor, class_idx=None, topk=10, detection_index=None):
        raise NotImplementedError

    def release(self):
        self.hooks.release()


class DetectorGradCAM(DetectorCAMBase):
    def __call__(self, input_tensor, class_idx=None, topk=10, detection_index=None):
        outputs = self.model(input_tensor)
        scalar_score, info = self._build_target(
            outputs, class_idx=class_idx, topk=topk, detection_index=detection_index
        )

        activations, gradients = self._forward_and_backward(input_tensor, scalar_score)

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = self._normalize_cam(cam)

        return cam, outputs, info


class DetectorGradCAMpp(DetectorCAMBase):
    def __call__(self, input_tensor, class_idx=None, topk=10, detection_index=None):
        outputs = self.model(input_tensor)
        scalar_score, info = self._build_target(
            outputs, class_idx=class_idx, topk=topk, detection_index=detection_index
        )

        activations, gradients = self._forward_and_backward(input_tensor, scalar_score)

        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)

        sum_activations_grad_3 = (activations * grad_3).sum(dim=(2, 3), keepdim=True)
        alpha_denom = 2 * grad_2 + sum_activations_grad_3
        alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = grad_2 / (alpha_denom + 1e-7)
        positive_gradients = F.relu(gradients)
        weights = (alpha * positive_gradients).sum(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = self._normalize_cam(cam)

        return cam, outputs, info
