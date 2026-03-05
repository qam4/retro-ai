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
    ):
        self._path = path
        self._fps = fps
        self._overlay = overlay
        self._writer: Optional[object] = None
        self._cv2 = None

        if not self.available():
            logger.warning("opencv-python not installed; " "video recording disabled")
            return

        import cv2

        self._cv2 = cv2

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
            fourcc = self._cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = self._cv2.VideoWriter(self._path, fourcc, self._fps, (w, h))

        out = frame.copy()
        if self._overlay and (reward != 0.0 or step != 0):
            text = f"R:{reward:.1f} S:{step}"
            self._cv2.putText(
                out,
                text,
                (5, 15),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
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
