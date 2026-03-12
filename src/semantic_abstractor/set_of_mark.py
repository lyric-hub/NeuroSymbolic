"""
Rendering Abstraction Layer for SoM (Set-of-Marks) Prompting.

This module provides a pluggable rendering system for visual grounding
in dense scenes. It is designed to:

- Separate rendering logic from detection/tracking.
- Support adaptive rendering strategies.
- Minimize visual noise for VLM grounding.
- Enable future extensions (graph overlays, occlusion reasoning, etc.).

Author: Your Project
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------


@dataclass
class TrackState:
    """
    Represents a single tracked object in a frame.

    Attributes:
        track_id (int): Unique tracking ID.
        box (np.ndarray): Bounding box in format [x1, y1, x2, y2].
        center (Tuple[int, int]): Centroid (cx, cy) derived from bounding box.
        velocity (Tuple[float, float]): Optional velocity vector.
        age (int): Number of frames object has persisted.
    """

    track_id: int
    box: np.ndarray
    center: Tuple[int, int]
    velocity: Tuple[float, float] = (0.0, 0.0)
    age: int = 0


@dataclass
class RenderContext:
    """
    Encapsulates the current rendering state.

    This context is updated each frame and passed to the renderer.

    Attributes:
        tracks (List[TrackState]): List of active track states.
        density (int): Number of active objects.
        timestamp (float): Frame timestamp.
    """

    tracks: List[TrackState] = field(default_factory=list)
    density: int = 0
    timestamp: float = 0.0

    def update(
        self, raw_tracks: np.ndarray, timestamp: Optional[float] = None
    ) -> None:
        """
        Updates the context from tracker output.

        Args:
            raw_tracks (np.ndarray): Array of shape (N, >=5) with format:
                [x1, y1, x2, y2, track_id, ...]
            timestamp (Optional[float]): Frame timestamp.
        """
        self.tracks.clear()
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.density = len(raw_tracks)

        for track_data in raw_tracks:
            x1, y1, x2, y2, track_id = track_data[:5]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            self.tracks.append(
                TrackState(
                    track_id=int(track_id),
                    box=np.array([x1, y1, x2, y2], dtype=int),
                    center=(cx, cy),
                )
            )


# ---------------------------------------------------------------------
# Base Renderer
# ---------------------------------------------------------------------


class BaseRenderer:
    """
    Abstract base class for rendering strategies.

    Subclasses must implement `render`.
    """

    def render(self, frame: np.ndarray, context: RenderContext) -> None:
        """
        Render overlays onto the frame.

        Args:
            frame (np.ndarray): BGR image frame.
            context (RenderContext): Scene state for current frame.

        Raises:
            NotImplementedError: If not overridden by subclass.
        """
        raise NotImplementedError("Subclasses must implement render().")


# ---------------------------------------------------------------------
# Ultra-Minimal Renderer
# ---------------------------------------------------------------------


class UltraMinimalRenderer(BaseRenderer):
    """
    Renders minimal SoM markers for dense scenes.

    Design:
    - Small centroid dot
    - Thin ID label
    - Neutral grayscale
    - No bounding boxes
    """

    def __init__(self, font_scale: float = 0.4) -> None:
        """
        Initializes the ultra-minimal renderer.

        Args:
            font_scale (float): Font size for ID labels.
        """
        self.font_scale = font_scale

    def render(self, frame: np.ndarray, context: RenderContext) -> None:
        """
        Render ultra-minimal markers.

        Args:
            frame (np.ndarray): BGR image frame.
            context (RenderContext): Overlays to draw.
        """
        for track in context.tracks:
            self._draw_track(frame, track)

    def _draw_track(self, frame: np.ndarray, track: TrackState) -> None:
        """
        Draw a single track marker.

        Args:
            frame (np.ndarray): The image frame.
            track (TrackState): The individual state to draw.
        """
        color = (200, 200, 200)  # soft gray
        cx, cy = track.center
        label = str(track.track_id)

        # centroid dot
        cv2.circle(frame, (cx, cy), 2, color, -1)

        # halo for contrast
        cv2.putText(
            frame,
            label,
            (cx + 4, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # main label
        cv2.putText(
            frame,
            label,
            (cx + 4, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            color,
            1,
            cv2.LINE_AA,
        )


# ---------------------------------------------------------------------
# Adaptive Renderer
# ---------------------------------------------------------------------


class AdaptiveRenderer(BaseRenderer):
    """
    Adaptive renderer that switches strategy based on scene density.

    Low density  -> corner ticks + centroid
    High density -> ultra-minimal mode
    """

    def __init__(self, density_threshold: int = 25) -> None:
        """
        Initializes the adaptive renderer.

        Args:
            density_threshold (int): Turn on minimal mode at this object count.
        """
        self.density_threshold = density_threshold
        self.minimal_renderer = UltraMinimalRenderer()

    def render(self, frame: np.ndarray, context: RenderContext) -> None:
        """
        Render using adaptive strategy.

        Args:
            frame (np.ndarray): BGR image frame.
            context (RenderContext): The context to track densities.
        """
        if context.density > self.density_threshold:
            self.minimal_renderer.render(frame, context)
        else:
            self._render_moderate(frame, context)

    def _render_moderate(
        self, frame: np.ndarray, context: RenderContext
    ) -> None:
        """
        Render moderate overlay style with light corner ticks.

        Args:
            frame (np.ndarray): The target image frame.
            context (RenderContext): Items to draw.
        """
        for track in context.tracks:
            x1, y1, x2, y2 = track.box
            cx, cy = track.center
            color = (180, 180, 180)
            tick = 6

            # centroid
            cv2.circle(frame, (cx, cy), 2, color, -1)

            # corner ticks (thin)
            cv2.line(frame, (x1, y1), (x1 + tick, y1), color, 1)
            cv2.line(frame, (x1, y1), (x1, y1 + tick), color, 1)
            cv2.line(frame, (x2, y1), (x2 - tick, y1), color, 1)
            cv2.line(frame, (x2, y1), (x2, y1 + tick), color, 1)

            cv2.putText(
                frame,
                str(track.track_id),
                (cx + 4, cy - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )