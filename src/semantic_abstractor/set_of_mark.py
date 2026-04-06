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
from typing import FrozenSet, List, Optional, Tuple

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
        zone_occupant_ids (Optional[FrozenSet[int]]): When set, only vehicles
            in this set receive SoM badges.  Vehicles outside the zone are
            rendered without an ID so the VLM cannot reference them.
            None means no zone filter (all tracks rendered).
    """

    tracks: List[TrackState] = field(default_factory=list)
    density: int = 0
    timestamp: float = 0.0
    zone_occupant_ids: Optional[FrozenSet[int]] = None

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
# Zone overlay — spotlight effect for VLM visual grounding
# ---------------------------------------------------------------------


def draw_zone_overlay(
    frame: np.ndarray,
    polygon: List[Tuple[float, float]],
    gates: Optional[List] = None,
    zone_id: str = "",
    outside_alpha: float = 0.35,
) -> None:
    """
    Apply a spotlight effect so the VLM visually focuses on the zone.

    Effect:
    - Everything **outside** the zone polygon is dimmed by ``outside_alpha``
      (0.35 = 65% darker), creating a vignette that draws VLM attention inward.
    - The zone boundary is drawn as a bright cyan polygon border (2 px).
    - Each gate is drawn as a yellow line with its name label at the midpoint.
    - The zone_id label is placed at the top-left corner of the polygon.

    This is called on the SoM frame **before** badge rendering so that badges
    appear on top of the overlay at full brightness.

    Args:
        frame:         BGR image array, modified in-place.
        polygon:       Zone polygon as [(x, y), ...] in pixel coordinates.
        gates:         Optional list of Gate-like objects with .name, .p1, .p2.
        zone_id:       Human-readable zone label drawn on the frame.
        outside_alpha: Brightness multiplier for the outside region (0–1).
    """
    if not polygon:
        return

    pts = np.array(polygon, dtype=np.int32)

    # --- Spotlight: dim outside the zone -----------------------------------
    # Build a single-channel mask: 255 inside polygon, 0 outside.
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # Multiply outside pixels by outside_alpha in-place (no copy needed).
    outside = mask == 0
    frame[outside] = (frame[outside].astype(np.float32) * outside_alpha).astype(np.uint8)

    # --- Zone border -------------------------------------------------------
    # Cyan (BGR: 255, 200, 0) — high contrast against both bright and dark scenes.
    ZONE_COLOR = (0, 220, 180)
    cv2.polylines(frame, [pts], isClosed=True, color=ZONE_COLOR, thickness=2)

    # --- Zone ID label -----------------------------------------------------
    if zone_id:
        lx = int(pts[:, 0].min())
        ly = int(pts[:, 1].min()) - 6
        ly = max(14, ly)  # clamp above top edge
        font = cv2.FONT_HERSHEY_SIMPLEX
        label = f"ZONE: {zone_id}"
        (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
        # Dark background pill for readability
        cv2.rectangle(frame, (lx - 2, ly - th - 3), (lx + tw + 2, ly + 2),
                      (0, 0, 0), -1)
        cv2.putText(frame, label, (lx, ly), font, 0.45, ZONE_COLOR, 1, cv2.LINE_AA)

    # --- Gate lines + labels -----------------------------------------------
    if gates:
        GATE_COLOR = (0, 220, 255)  # yellow-ish
        for gate in gates:
            p1 = (int(gate.p1[0]), int(gate.p1[1]))
            p2 = (int(gate.p2[0]), int(gate.p2[1]))
            cv2.line(frame, p1, p2, GATE_COLOR, 2)
            # Arrow tips to show directionality
            cv2.arrowedLine(frame, p1, p2, GATE_COLOR, 1, tipLength=0.15)
            # Gate name at midpoint with dark background
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(gate.name, font, 0.4, 1)
            cv2.rectangle(frame,
                          (mid[0] - 2, mid[1] - th - 2),
                          (mid[0] + tw + 2, mid[1] + 2),
                          (0, 0, 0), -1)
            cv2.putText(frame, gate.name, mid, font, 0.4, GATE_COLOR, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------
# Ultra-Minimal Renderer
# ---------------------------------------------------------------------


def _draw_som_badge(
    frame: np.ndarray,
    cx: int,
    cy: int,
    label: str,
    font_scale: float = 0.55,
) -> None:
    """
    Draw a high-contrast SoM badge (filled coloured box + white text) so that
    VLMs can reliably read the track ID from the image.

    A unique colour per ID is derived from the track ID so repeated badges are
    visually distinguishable even in dense scenes.
    """
    font      = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    pad       = 3

    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Deterministic per-ID colour (hue derived from ID, full saturation/value)
    hue   = int(label) * 37 % 180          # spread IDs across hue wheel
    hsv   = np.array([[[hue, 220, 200]]], dtype=np.uint8)
    bgr   = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()

    # Centroid dot
    cv2.circle(frame, (cx, cy), 4, bgr, -1)
    cv2.circle(frame, (cx, cy), 4, (255, 255, 255), 1)

    # Badge background
    bx1, by1 = cx + 6, cy - th - pad * 2
    bx2, by2 = cx + 6 + tw + pad * 2, cy
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), bgr, -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 1)

    # White text on coloured badge
    cv2.putText(
        frame, label,
        (bx1 + pad, by2 - pad),
        font, font_scale,
        (255, 255, 255), thickness, cv2.LINE_AA,
    )


class UltraMinimalRenderer(BaseRenderer):
    """
    Renders SoM badges for dense scenes.

    Design:
    - Coloured filled badge with white text for VLM readability
    - Centroid dot with white outline
    - No bounding boxes (minimises visual clutter)
    """

    def __init__(self, font_scale: float = 0.55) -> None:
        self.font_scale = font_scale

    def render(self, frame: np.ndarray, context: RenderContext) -> None:
        for track in context.tracks:
            # Zone filter: skip badges for vehicles outside the zone so the
            # VLM cannot reference or reason about out-of-zone vehicles.
            if (context.zone_occupant_ids is not None
                    and track.track_id not in context.zone_occupant_ids):
                continue
            self._draw_track(frame, track)

    def _draw_track(self, frame: np.ndarray, track: TrackState) -> None:
        cx, cy = track.center
        _draw_som_badge(frame, cx, cy, str(track.track_id), self.font_scale)


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

        Zone filter is propagated to both paths: when context.zone_occupant_ids
        is set, only in-zone vehicles receive SoM badges in either mode.

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
            # Zone filter: skip out-of-zone vehicles entirely
            if (context.zone_occupant_ids is not None
                    and track.track_id not in context.zone_occupant_ids):
                continue

            x1, y1, x2, y2 = track.box
            cx, cy = track.center

            # Derive per-ID colour (same palette as badge)
            hue   = int(track.track_id) * 37 % 180
            hsv   = np.array([[[hue, 220, 200]]], dtype=np.uint8)
            color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()
            tick  = 8

            # Corner ticks in matching colour
            cv2.line(frame, (x1, y1), (x1 + tick, y1), color, 2)
            cv2.line(frame, (x1, y1), (x1, y1 + tick), color, 2)
            cv2.line(frame, (x2, y1), (x2 - tick, y1), color, 2)
            cv2.line(frame, (x2, y1), (x2, y1 + tick), color, 2)

            # High-contrast SoM badge
            _draw_som_badge(frame, cx, cy, str(track.track_id))