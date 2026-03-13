# detector.py
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

# Third-party imports
import numpy as np
import torch
from PIL import Image

# Local imports
from src.utils import get_device

# --- Constants & Types ---
PathLike = Union[str, Path]
DeviceArg = Optional[Union[str, int, torch.device]]


def _resolve_device(device_arg: DeviceArg) -> torch.device:
    """
    Resolves the provided device argument to a torch.device.

    Args:
        device_arg (DeviceArg): A string, integer, or torch.device.

    Returns:
        torch.device: The resolved PyTorch device.
    """
    if device_arg is None:
        return get_device()

    if isinstance(device_arg, torch.device):
        return device_arg

    if isinstance(device_arg, str) and device_arg.strip().lower() == "auto":
        return get_device()

    if isinstance(device_arg, int):
        # Maps int e.g., 0 to 'cuda:0'
        return torch.device(f"cuda:{device_arg}")

    # Handle strings like 'cpu', 'cuda', 'cuda:0'
    device_str = str(device_arg).strip().lower()

    if device_str == "cpu":
        return torch.device("cpu")

    if device_str in ("cuda", "gpu"):
        # Default to index 0 if specific index not provided
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    return torch.device(device_str)


def _norm_label(label: str) -> str:
    """
    Normalizes a label string by removing common articles.

    Args:
        label (str): The input label.

    Returns:
        str: The normalized label string.
    """
    normalized = label.strip().lower()
    for prefix in ("a ", "an ", "the "):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    return normalized


def _to_pil(image_source: Any) -> Image.Image:
    """
    Converts various image sources (str, Path, array, tensor) to
    a PIL RGB Image.

    Args:
        image_source (Any): The input image as a path, array, tensor,
            or PIL Image.

    Returns:
        Image.Image: An RGB representation of the image.

    Raises:
        TypeError: If the input image type is not supported.
    """
    if isinstance(image_source, (str, Path)):
        return Image.open(image_source).convert("RGB")

    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")

    if isinstance(image_source, np.ndarray):
        # Handle HWC numpy array
        if image_source.ndim == 3 and image_source.shape[2] in (1, 3, 4):
            return Image.fromarray(image_source[..., :3]).convert("RGB")
        # Handle Grayscale
        if image_source.ndim == 2:
            return Image.fromarray(image_source).convert("RGB")

    if isinstance(image_source, torch.Tensor):
        tensor = image_source.detach().cpu()
        # Convert CHW to HWC
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        if tensor.max() <= 1.0:
            tensor = (tensor * 255.0).clamp(0, 255)
        tensor_np = tensor.to(torch.uint8).numpy()
        return Image.fromarray(tensor_np[..., :3]).convert("RGB")

    raise TypeError(f"Unsupported image type: {type(image_source)}")


def _is_gdino_spec(spec: str) -> bool:
    """
    Checks if the model specification string refers to GroundingDINO.

    Args:
        spec (str): The model path or specification.

    Returns:
        bool: True if the spec matches known patterns for GroundingDINO.
    """
    spec_lower = spec.strip().lower()
    return (
        spec_lower.startswith(("gdino:", "groundingdino:"))
        or spec_lower in {"gdino", "groundingdino"}
        or "grounding-dino" in spec_lower
    )


def _gdino_model_id_from_spec(spec: str) -> str:
    """
    Extracts the HuggingFace model ID from a GroundingDINO spec string.

    Args:
        spec (str): The raw spec string.

    Returns:
        str: The canonical Hugging Face repository ID.
    """
    spec_clean = spec.strip()
    if spec_clean.lower() in {"gdino", "groundingdino"}:
        return "IDEA-Research/grounding-dino-base"

    if ":" in spec_clean:
        _, rest = spec_clean.split(":", 1)
        return rest.strip() or "IDEA-Research/grounding-dino-base"

    return spec_clean


@dataclass(frozen=True)
class DetectDefaults:
    """Configuration defaults for detectors."""

    device: DeviceArg = None
    half: bool = False
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.7


class Detector:
    """Base class for object detectors."""

    def predict(
        self, source: Any, *, vocab: Optional[Sequence[str]] = None, **kwargs
    ) -> Any:
        """
        Runs object detection on the given source.

        Args:
            source (Any): The image or frame to detect on.
            vocab (Optional[Sequence[str]]): An optional list of classes
                to look for.
            **kwargs: Additional parameters for the predictor
                (like conf thresholds).

        Returns:
            Any: The raw predictions format dependent on the underlying model.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Allows instances to be used directly as Callables."""
        return self.predict(*args, **kwargs)


class YoloDetector(Detector):
    """
    Wrapper for Ultralytics YOLO models.

    Allows loading models and applying predictions with open-vocabulary
    support.
    """

    def __init__(self, model: Any, defaults: DetectDefaults) -> None:
        """
        Initializes the YOLO detector.

        Args:
            model (Any): An instantiated Ultralytics YOLO model.
            defaults (DetectDefaults): The default prediction constants.
        """
        self.model = model
        self.defaults = defaults
        self._current_vocab: Optional[Tuple[str, ...]] = None

    def predict(
        self,
        source: Any,
        *,
        vocab: Optional[Sequence[str]] = None,
        classes: Optional[Sequence[int]] = None,
        **kwargs,
    ) -> Any:
        """
        Predicts objects using the loaded YOLO model.

        Resolves configuration from kwargs or falls back to init defaults.
        """
        # Resolve parameters falling back to defaults
        imgsz = kwargs.pop("imgsz", self.defaults.imgsz)
        conf = kwargs.pop("conf", self.defaults.conf)
        iou = kwargs.pop("iou", self.defaults.iou)
        half = kwargs.pop("half", self.defaults.half)
        device_arg = kwargs.pop("device", self.defaults.device)

        resolved_device = _resolve_device(device_arg)

        # Prepare vocabulary if present (Open Vocabulary YOLO)
        if vocab is not None:
            vocab_tuple = tuple(vocab)
            if vocab_tuple != self._current_vocab:
                self._apply_vocab(vocab_tuple)
                self._current_vocab = vocab_tuple

        return self.model.predict(
            source=source,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            half=half,
            device=resolved_device.index
            if resolved_device.type == "cuda" else "cpu",
            classes=list(classes) if classes is not None else None,
            agnostic_nms=True,  # Suppress duplicate detections across classes
            verbose=False,
        )

    def _apply_vocab(self, vocab: Sequence[str]) -> None:
        """
        Helper to set classes on supported YOLO models.

        Args:
            vocab (Sequence[str]): Class labels.
        """
        vocab_list = list(vocab)
        # Try different API methods supported by various YOLO versions
        try:
            self.model.set_classes(vocab_list)
            return
        except (AttributeError, TypeError):
            pass

        if hasattr(self.model, "get_text_pe"):
            pe = self.model.get_text_pe(vocab_list)
            if hasattr(self.model, "set_classes"):
                self.model.set_classes(vocab_list, pe)
                return

        if hasattr(self.model, "set_vocab"):
            self.model.set_vocab(vocab_list)
            return

        logging.warning(
            "This YOLO model does not seem to support open-vocabulary prompts."
        )


class GroundingDinoDetector(Detector):
    """
    Wrapper for Hugging Face GroundingDINO models.

    Performs zero-shot predictions utilizing a textual prompt (vocabulary).
    """

    def __init__(
        self,
        model_id: str,
        defaults: DetectDefaults,
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
    ) -> None:
        """
        Initializes the GroundingDINO detector.

        Args:
            model_id (str): The HF repository string
                for GroundingDINO.
            defaults (DetectDefaults): Fallback configuration for device
                and precision.
            box_threshold (float): Filter bounding boxes below this threshold.
            text_threshold (float): Filter class text thresholds.
        """
        self.model_id = model_id
        self.defaults = defaults
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Lazy import for heavy dependencies only when instantiated
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id)

        # Device setup
        self._device = _resolve_device(defaults.device)
        self.model.to(self._device)

        if defaults.half and self._device.type == "cuda":
            self.model.half()

        self.model.eval()

    def predict(
        self, source: Any, *, vocab: Optional[Sequence[str]] = None, **kwargs
    ) -> List[Any]:
        """
        Predicts objects using GroundingDINO and the provided textual vocab.
        """
        if not vocab:
            raise ValueError(
                "GroundingDINO requires vocab=[...] (candidate labels).")

        from ultralytics.engine.results import Boxes, Results

        box_thr = float(kwargs.pop("box_threshold", self.box_threshold))
        text_thr = float(kwargs.pop("text_threshold", self.text_threshold))
        classes_filter = kwargs.get("classes", None)

        items = list(source) if isinstance(source, (list, tuple)) else [source]
        pil_images = [_to_pil(item) for item in items]

        names_map = {i: v for i, v in enumerate(vocab)}
        vocab_lookup = {_norm_label(v): i for i, v in enumerate(vocab)}

        # FIX 1: Explicitly format prompts with dots ("cat . dog .")
        prompt_text = " . ".join(vocab)
        if not prompt_text.endswith("."):
            prompt_text += " ."

        # Replicate for batch
        text_prompts = [prompt_text] * len(pil_images)

        inputs = self.processor(
            images=pil_images, text=text_prompts, return_tensors="pt"
        )
        inputs = {
            k: v.to(self._device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)

        target_sizes = [img.size[::-1] for img in pil_images]

        raw_results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=box_thr,
            text_threshold=text_thr,
            target_sizes=target_sizes,
        )

        final_results = []
        for i, res in enumerate(raw_results):
            boxes_xyxy = res.get("boxes", torch.empty(
                (0, 4))).detach().cpu().float()
            scores = res.get("scores", torch.empty((0,))
                             ).detach().cpu().float()
            labels_text = [str(lbl) for lbl in res.get("labels", [])]

            # FIX 2: Safer lookup with filter - default to -1 for unknown
            # labels
            cls_ids = []
            keep_indices = []
            for idx, lbl in enumerate(labels_text):
                norm = _norm_label(lbl)
                cid = vocab_lookup.get(norm, -1)

                # Check mapping and optional classes filter
                if cid != -1:
                    if classes_filter is None or cid in classes_filter:
                        cls_ids.append(cid)
                        keep_indices.append(idx)

            if keep_indices:
                # Filter tensors to only keep valid detections
                keep = torch.tensor(keep_indices, dtype=torch.long)
                boxes_xyxy = boxes_xyxy[keep]
                scores = scores[keep]
                cls_tensor = torch.tensor(cls_ids, dtype=torch.float32)

                pred = torch.cat(
                    [
                        boxes_xyxy,
                        scores.unsqueeze(1),
                        cls_tensor.unsqueeze(1)
                    ],
                    dim=1,
                )
            else:
                pred = torch.zeros((0, 6), dtype=torch.float32)

            orig_rgb = np.asarray(pil_images[i])
            orig_bgr = orig_rgb[..., ::-1].copy()

            result = Results(orig_img=orig_bgr, path="", names=names_map)
            result.boxes = Boxes(pred, orig_bgr.shape[:2])
            final_results.append(result)

        return final_results


@lru_cache(maxsize=32)
def _load_yolo_model(weights: str) -> Any:
    """
    Loads an Ultralytics YOLO model from a weight file path.

    Attempts to prefer TensorRT (.engine) over raw PyTorch (.pt) if available.

    Args:
        weights (str): Path to the model file.

    Returns:
        Any: The loaded YOLO model.
    """
    from ultralytics import YOLO

    weight_path = Path(weights).expanduser().resolve()

    def _tensorrt_available() -> bool:
        try:
            import tensorrt  # noqa: F401
            return True
        except ImportError:
            return False

    if weight_path.suffix.lower() == ".engine":
        if not _tensorrt_available():
            raise RuntimeError(
                f"Model {weight_path} is a TensorRT engine but the "
                "'tensorrt' package is not installed. "
                "Install it or pass the .pt weights path instead."
            )
        logging.info(f"Loading TensorRT engine: {weight_path}")
        return YOLO(str(weight_path), task="detect")

    # If .pt given, prefer engine sibling only when TensorRT is available
    engine_sibling = weight_path.with_suffix(".engine")

    if engine_sibling.exists():
        if _tensorrt_available():
            logging.info(f"Found TensorRT engine: {engine_sibling}")
            return YOLO(str(engine_sibling), task="detect")
        logging.info(
            f"TensorRT engine found at {engine_sibling} but 'tensorrt' is "
            "not installed — falling back to PyTorch model."
        )

    return YOLO(str(weight_path), task="detect")


def load_detector(
    spec: str,
    *,
    device: DeviceArg = None,
    half: bool = False,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
    box_threshold: float = 0.4,
    text_threshold: float = 0.3,
) -> Detector:
    """
    Factory function to load a detector based on the specification string.

    Args:
        spec (str): Model path (e.g., 'yolov8n.pt') or
            ID (e.g., 'gdino:idea...').
        device (DeviceArg): Device to load model on
            ('cuda', 'cpu', 0, or None).
        half (bool): Use FP16 if true and on CUDA.
        imgsz (int): Inference image size.
        conf (float): Object confidence threshold.
        iou (float): IOU threshold for NMS.
        box_threshold (float): (GDINO only) Object bounding box threshold.
        text_threshold (float): (GDINO only) Text grouping threshold.

    Returns:
        Detector: An instance of a Detector subclass.
    """
    defaults = DetectDefaults(device=device, half=half,
                              imgsz=imgsz, conf=conf, iou=iou)

    if _is_gdino_spec(spec):
        model_id = _gdino_model_id_from_spec(spec)
        return GroundingDinoDetector(
            model_id,
            defaults,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

    model = _load_yolo_model(spec)
    return YoloDetector(model, defaults)