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
            raise RuntimeError(
                "Video recording requested but opencv-python is not installed. "
                "Install it with: pip install opencv-python"
            )

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
        action=None,
        step_reward: float = 0.0,
    ) -> None:
        """Write a frame. Initializes writer on first call.

        Parameters
        ----------
        reward : float
            Cumulative reward so far.
        step_reward : float
            Reward received on this specific step.
        action : list or int or None
            For multi-discrete: [up, down, left, right, fire] binary list.
            For discrete: integer action index. None to skip action overlay.
        """
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
            if abs(step_reward) > 0.001:
                text += (
                    f"  +{step_reward:.1f}"
                    if step_reward > 0
                    else f"  {step_reward:.1f}"
                )
            font_scale = max(0.4, 0.4 * self._scale)
            thickness = max(1, self._scale)
            self._cv2.putText(
                out,
                text,
                (4, 14 * max(1, self._scale)),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
            )

        if self._overlay and action is not None:
            self._draw_action_overlay(out, action)

        # OpenCV expects BGR
        bgr = self._cv2.cvtColor(out, self._cv2.COLOR_RGB2BGR)
        self._writer.write(bgr)

    def _draw_action_overlay(self, frame: np.ndarray, action) -> None:
        """Draw a joystick + fire button indicator in the bottom-right corner."""
        cv2 = self._cv2
        h, w = frame.shape[:2]

        # Unwrap 0-dim numpy arrays (e.g. from model.predict())
        if isinstance(action, np.ndarray) and action.ndim == 0:
            action = action.item()

        # Parse action into direction bools
        if (
            isinstance(action, (list, np.ndarray))
            and np.ndim(action) > 0
            and len(action) >= 5
        ):
            up, down, left, right, fire = (bool(a) for a in action[:5])
        elif (
            isinstance(action, (list, np.ndarray))
            and np.ndim(action) > 0
            and len(action) == 3
        ):
            # Joystick mode: [vertical(3), horizontal(3), fire(2)]
            # vertical: 0=neutral, 1=up, 2=down
            # horizontal: 0=neutral, 1=right, 2=left
            vert, horiz, btn = (int(a) for a in action[:3])
            up = vert == 1
            down = vert == 2
            right = horiz == 1
            left = horiz == 2
            fire = btn == 1
        elif isinstance(action, (int, np.integer)):
            # Discrete action mapping
            up = action in (1, 6)
            down = action in (2, 7)
            left = action in (3, 8)
            right = action in (4, 9)
            fire = action in (5, 6, 7, 8, 9)
        else:
            return

        # Layout: small joystick cross + fire circle in bottom-right
        r = max(4, min(w, h) // 30)  # dot radius scales with frame
        gap = r * 3  # spacing between dots
        cx = w - gap * 3  # center of cross (more room for fire button)
        cy = h - gap * 2
        # Colors in RGB (frame is RGB, converted to BGR after overlay)
        off_color = (80, 80, 80)  # dark gray = inactive
        on_color = (255, 255, 0)  # yellow = active direction
        fire_off = (80, 80, 80)
        fire_on = (255, 50, 50)  # red = fire active

        # Draw direction dots: up, down, left, right
        positions = {
            "up": (cx, cy - gap),
            "down": (cx, cy + gap),
            "left": (cx - gap, cy),
            "right": (cx + gap, cy),
        }
        states = {"up": up, "down": down, "left": left, "right": right}
        for name, pos in positions.items():
            color = on_color if states[name] else off_color
            cv2.circle(frame, pos, r, color, -1)

        # Center dot (neutral indicator)
        cv2.circle(frame, (cx, cy), r // 2, (120, 120, 120), -1)

        # Fire button — to the right of the cross
        fire_pos = (cx + gap * 2, cy)
        color = fire_on if fire else fire_off
        cv2.circle(frame, fire_pos, int(r * 1.3), color, -1)

    def close(self) -> None:
        """Release the video writer and re-encode to H.264 if ffmpeg is available."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self._reencode_h264()

    @staticmethod
    def _find_ffmpeg() -> Optional[str]:
        """Locate ffmpeg: system PATH first, then imageio-ffmpeg fallback."""
        import shutil

        path = shutil.which("ffmpeg")
        if path:
            return path
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            return None

    def _reencode_h264(self) -> None:
        """Re-encode mp4v to H.264 using ffmpeg for browser compatibility."""
        import os
        import subprocess

        ffmpeg = self._find_ffmpeg()
        if not ffmpeg or not self._path.endswith(".mp4"):
            return

        tmp = self._path + ".tmp.mp4"
        try:
            result = subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    self._path,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    tmp,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                os.replace(tmp, self._path)
            else:
                logger.warning("ffmpeg re-encode failed: %s", result.stderr[:200])
                if os.path.exists(tmp):
                    os.unlink(tmp)
        except Exception as e:
            logger.warning("ffmpeg re-encode error: %s", e)
            if os.path.exists(tmp):
                os.unlink(tmp)

    @staticmethod
    def available() -> bool:
        """Check if OpenCV is importable."""
        try:
            import cv2  # noqa: F401

            return True
        except ImportError:
            return False
