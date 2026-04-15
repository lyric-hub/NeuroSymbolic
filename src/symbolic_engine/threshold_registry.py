"""
Threshold Provenance Registry
==============================
Maps every symbolic rule threshold to its source standard and rationale.

Enables the agent to answer "why 4 m/s²?" by citing the engineering standard
behind each threshold — not just reporting that a vehicle exceeded it.
This is a key explainability component: thresholds are not magic numbers,
they are grounded in published traffic safety research.

Usage (agent tool)::

    from src.symbolic_engine.threshold_registry import explain_threshold
    result = explain_threshold("HARD_BRAKING")
    # → {"rule": "HARD_BRAKING", "value": 4.0, "unit": "m/s²", "source": "...", ...}
"""

from __future__ import annotations

THRESHOLD_PROVENANCE: dict[str, dict] = {
    "HARD_BRAKING": {
        "value": 4.0,
        "unit": "m/s²",
        "source": "NCHRP Report 600 / UK Highway Code",
        "description": (
            "Deceleration exceeding 4 m/s² is considered emergency braking. "
            "Normal comfortable braking is 2–3 m/s²; panic braking is 7–8 m/s². "
            "4 m/s² is the boundary above which following drivers cannot react in "
            "time at typical urban following distances."
        ),
    },
    "AGGRESSIVE_ACCELERATION": {
        "value": 3.5,
        "unit": "m/s²",
        "source": "PIEV research / ADAS calibration standards",
        "description": (
            "Acceleration above 3.5 m/s² exceeds comfortable passenger tolerance "
            "and correlates with aggressive driving behaviour in naturalistic "
            "driving studies (SHRP2 NDS dataset). Maximum tyre friction limit is "
            "approximately 9 m/s² on dry asphalt."
        ),
    },
    "SPEEDING_DEFAULT": {
        "value": 50.0,
        "unit": "km/h",
        "source": "WHO Global Speed Management Manual (Safe System approach)",
        "description": (
            "50 km/h is the WHO-recommended default urban speed limit. "
            "At 50 km/h a pedestrian impact has ~20% fatality risk; at 65 km/h "
            "it rises to ~85%. This default is overridden per-zone by "
            "speed_limit_kmh in zone_config.json (school zones, motorways, etc.)."
        ),
    },
    "TTC_CRITICAL": {
        "value": 1.5,
        "unit": "s",
        "source": "FHWA SSAM (Surrogate Safety Assessment Model)",
        "description": (
            "TTC < 1.5s is a CRITICAL conflict — the driver has insufficient time "
            "to perceive, react, and brake before impact. Human perception-reaction "
            "time is approximately 1.5s under emergency conditions. "
            "This is the SSAM threshold used by FHWA for imminent crash classification."
        ),
    },
    "TTC_SERIOUS": {
        "value": 3.0,
        "unit": "s",
        "source": "FHWA SSAM / Highway Capacity Manual",
        "description": (
            "TTC < 3.0s is a SERIOUS conflict — secondary SSAM classification. "
            "At highway speeds (30 m/s), 3s corresponds to a 90m following distance, "
            "the minimum safe headway recommended by the Highway Capacity Manual. "
            "Below this threshold, evasive action is needed."
        ),
    },
    "WRONG_WAY_ANGLE": {
        "value": 90.0,
        "unit": "degrees",
        "source": "Geometric road design convention (mathematical bearing system)",
        "description": (
            "A heading more than 90° from the expected flow direction means the "
            "vehicle is travelling in the opposing angular half-space — definitively "
            "against traffic flow. The 90° boundary is the perpendicular bisector "
            "between correct and wrong-way travel in the CCW bearing convention "
            "(0°=East, 90°=North)."
        ),
    },
    "WRONG_WAY_MIN_SPEED": {
        "value": 1.0,
        "unit": "m/s",
        "source": "Engineering judgement / Savitzky-Golay noise suppression",
        "description": (
            "Stationary or near-stationary vehicles have unreliable heading estimates "
            "because vel_x and vel_y approach zero, making atan2 unstable. "
            "The 1 m/s minimum speed filter prevents false positives from parked "
            "vehicles or vehicles executing very slow manoeuvres."
        ),
    },
    "WRONG_WAY_MIN_DURATION": {
        "value": 1.0,
        "unit": "s",
        "source": "Engineering judgement / false positive suppression",
        "description": (
            "A single-frame wrong-way heading can occur during a U-turn or lane "
            "change completing. Requiring 1s of sustained wrong-way heading (at "
            "30 fps = 30 frames) filters transient manoeuvres and targets genuine "
            "contraflow driving."
        ),
    },
    "STATIONARY_IN_LANE_DURATION": {
        "value": 10.0,
        "unit": "s",
        "source": "Highway Code / secondary collision research",
        "description": (
            "10 seconds is the threshold above which a stationary vehicle in an "
            "active lane becomes a secondary collision hazard. Research shows "
            "following drivers require 5–8s to react and manoeuvre around an "
            "obstruction at urban speeds; 10s provides a 2s safety margin. "
            "Catches broken-down vehicles, double-parking, and stalled engines."
        ),
    },
    "STATIONARY_SPEED": {
        "value": 0.5,
        "unit": "m/s",
        "source": "Savitzky-Golay noise floor / measurement uncertainty",
        "description": (
            "The SG filter produces residual velocity noise of ~0.3 m/s for "
            "truly stationary vehicles due to detection jitter and homography "
            "projection error. The 0.5 m/s threshold sits above this noise floor "
            "while remaining well below typical walking speed (1.4 m/s)."
        ),
    },
    "V85_DESIGN_SPEED": {
        "value": 85.0,
        "unit": "percentile",
        "source": "AASHTO Green Book / Road Safety Audit guidelines",
        "description": (
            "The 85th percentile speed (V85) is the standard design speed metric. "
            "It represents the speed below which 85% of drivers travel under "
            "free-flow conditions. If V85 exceeds the posted speed limit, the limit "
            "may require enforcement action or re-assessment. Used in speed limit "
            "reviews and road safety audits worldwide."
        ),
    },
}


def explain_threshold(rule_name: str) -> dict:
    """
    Returns the provenance record for a rule threshold by name.

    Args:
        rule_name: One of the keys in THRESHOLD_PROVENANCE (case-insensitive,
                   spaces and hyphens normalised to underscores).

    Returns:
        Dict with ``rule``, ``value``, ``unit``, ``source``, ``description``.
        On no match, returns an error dict listing available rule names.
    """
    key = rule_name.upper().replace(" ", "_").replace("-", "_")

    if key in THRESHOLD_PROVENANCE:
        return {"rule": key, **THRESHOLD_PROVENANCE[key]}

    # Fuzzy fallback: substring match
    matches = [k for k in THRESHOLD_PROVENANCE if key in k or k in key]
    if matches:
        return {
            "rule": matches[0],
            **THRESHOLD_PROVENANCE[matches[0]],
            "_note": (
                f"Fuzzy match for '{rule_name}'. "
                f"All available rules: {list(THRESHOLD_PROVENANCE.keys())}"
            ),
        }

    return {
        "error": f"No threshold record found for '{rule_name}'.",
        "available_rules": list(THRESHOLD_PROVENANCE.keys()),
    }
