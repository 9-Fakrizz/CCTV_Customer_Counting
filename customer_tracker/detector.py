from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as e:  # pragma: no cover
    YOLO = None  # type: ignore
    _import_error = e
else:
    _import_error = None


@dataclass
class Detection:
    xyxy: Tuple[float, float, float, float]
    conf: float
    head_xy: Optional[Tuple[float, float]] = None
    # Pose keypoints in original pixel coords: [(x,y,conf), ...]
    kpts: Optional[List[Tuple[float, float, float]]] = None


class PersonDetector:
    """YOLO wrapper: person class only, optional max-side resize."""

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        conf: float = 0.35,
        imgsz: int = 640,
        device: Optional[str] = None,
        iou: float = 0.5,
        max_det: int = 50,
    ) -> None:
        if _import_error is not None:
            raise RuntimeError(
                "ultralytics is required. pip install ultralytics"
            ) from _import_error
        self.model = YOLO(model_name)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.iou = float(iou)
        self.max_det = int(max_det)

    def detect_persons(
        self,
        frame_bgr: np.ndarray,
        scale_back: Tuple[float, float],
    ) -> List[Detection]:
        """Run on `frame_bgr`; multiply boxes by scale_back (sx, sy) to original coords."""
        sx, sy = scale_back
        results = self.model.predict(
            frame_bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            iou=self.iou,
            max_det=self.max_det,
            verbose=False,
            device=self.device,
        )
        out: List[Detection] = []
        if not results:
            return out
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return out
        boxes = r0.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        # Optional pose keypoints (if model is a pose model)
        kpt_xy = None
        kpt_conf = None
        if getattr(r0, "keypoints", None) is not None and r0.keypoints is not None:
            try:
                kpt_xy = r0.keypoints.xy.cpu().numpy()  # (n, k, 2)
                if getattr(r0.keypoints, "conf", None) is not None:
                    kpt_conf = r0.keypoints.conf.cpu().numpy()  # (n, k)
            except Exception:
                kpt_xy = None
                kpt_conf = None
        for i in range(len(xyxy)):
            if clss[i] != 0:
                continue
            x1, y1, x2, y2 = xyxy[i]
            head_xy: Optional[Tuple[float, float]] = None
            kpts: Optional[List[Tuple[float, float, float]]] = None
            # COCO keypoints: 0=nose, 1/2=eyes, 3/4=ears.
            # Faces are often partially occluded; pick the best available head keypoint.
            if kpt_xy is not None and i < len(kpt_xy) and kpt_xy.shape[1] > 0:
                head_idxs = [0, 1, 2, 3, 4]
                best: Optional[Tuple[float, float, float]] = None
                for hi in head_idxs:
                    if hi >= int(kpt_xy.shape[1]):
                        continue
                    hx, hy = kpt_xy[i][hi]
                    hc = float(kpt_conf[i][hi]) if kpt_conf is not None else 1.0
                    if hc < 0.05:
                        continue
                    if best is None or hc > best[2]:
                        best = (float(hx), float(hy), hc)
                if best is not None:
                    head_xy = (float(best[0] * sx), float(best[1] * sy))
                # keep all landmarks for preview/debug overlay
                k = int(kpt_xy.shape[1])
                kpts = []
                for j in range(k):
                    px, py = kpt_xy[i][j]
                    c = float(kpt_conf[i][j]) if kpt_conf is not None else 1.0
                    kpts.append((float(px * sx), float(py * sy), c))
            out.append(
                Detection(
                    xyxy=(
                        float(x1 * sx),
                        float(y1 * sy),
                        float(x2 * sx),
                        float(y2 * sy),
                    ),
                    conf=float(confs[i]),
                    head_xy=head_xy,
                    kpts=kpts,
                )
            )
        return out


def resize_for_inference(
    frame: np.ndarray, max_width: int
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Return resized frame and (sx, sy) to map model coords -> original."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, (1.0, 1.0)
    scale = max_width / float(w)
    new_w = max_width
    new_h = int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h))
    sx = w / float(new_w)
    sy = h / float(new_h)
    return resized, (sx, sy)
