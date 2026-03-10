"""Optional MP4 video recorder with graceful degradation."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class VideoRecorder:
    """Record frames to MP4. No-op if OpenCV is unavailable."""

    def __init__(
        self,
        path: str,
        fps: float = 60.0,
        overlay: bool = False,
        scale: int = 3,
        aspect_ratio: Optional[str] = None,
    ):
        self._path = path
        self._fps = fps
        self._overlay = overlay
        self._scale = max(1, scale)
        self._aspect_ratio = aspect_ratio
        self._writer: Optional[object] = None
        self._cv2 = None

        if not self.available():
            logger.warning("opencv-python not installed; " "video recording disabled")
            return

        import cv2

        self._cv2 = cv2

    def _compute_output_size(self, h: int, w: int) -> tuple:
        """Compute output (width, height) based on scale and aspect ratio.

        When aspect_ratio is set, the frame is stretched to match the target
        ratio before applying the integer scale factor. For example, a 160×240
        frame with aspect_ratio="4:3" becomes 320×240 (width doubled), then
        scaled by the scale factor.
        """
        if self._aspect_ratio:
            parts = self._aspect_ratio.split(":")
            if len(parts) == 2:
                ar_w, ar_h = float(parts[0]), float(parts[1])
                # Compute the width needed to achieve the target aspect ratio
                # at the current height, then apply scale
                target_w = int(h * ar_w / ar_h)
                return (target_w * self._scale, h * self._scale)
        return (w * self._scale, h * self._scale)

    def add_frame(
        self,
        frame: np.ndarray,
        reward: float = 0.0,
        step: int = 0,
    ) -> None:
        """Write a frame. Initializes writer on first call."""
        if self._cv2 is None:
            return

        # Handle grayscale or stacked frames
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        elif frame.shape[-1] > 3:
            # Stacked frames: take last 3 channels or last 1
            frame = frame[..., -3:]

        if self._writer is None:
            h, w = frame.shape[:2]
            out_w, out_h = self._compute_output_size(h, w)
            fourcc = self._cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = self._cv2.VideoWriter(
                self._path, fourcc, self._fps, (out_w, out_h)
            )

        # Resize to output dimensions (handles both scale and aspect ratio)
        h, w = frame.shape[:2]
        out_w, out_h = self._compute_output_size(h, w)
        if out_w != w or out_h != h:
            frame = self._cv2.resize(
                frame,
                (out_w, out_h),
                interpolation=self._cv2.INTER_NEAREST,
            )

        out = frame.copy()
        if self._overlay and (reward != 0.0 or step != 0):
            text = f"R:{reward:.0f}  Step:{step}"
            font_scale = 0.5 * (self._scale / 2)
            thickness = max(1, self._scale // 2)
            self._cv2.putText(
                out,
                text,
                (8, 20 * max(1, self._scale // 2)),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
            )

        # OpenCV expects BGR
        bgr = self._cv2.cvtColor(out, self._cv2.COLOR_RGB2BGR)
        self._writer.write(bgr)

    def close(self) -> None:
        """Release the video writer."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    @staticmethod
    def available() -> bool:
        """Check if OpenCV is importable."""
        try:
            import cv2  # noqa: F401

            return True
        except ImportError:
            return False
