import math
from typing import Optional

import numpy as np

from .candidate import GraspCandidate
from .evaluator import ThreeJawEvaluator
from .factory import EvaluatorFactory


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


@EvaluatorFactory.register("chick")
class ChickEvaluator(ThreeJawEvaluator):
    """
    Chick/UFO-like grabbing evaluator with measurable, physics-oriented terms.

    Measurable inputs used:
    - mask geometry (centroid, local support, neck-zone contacts)
    - depth map around center/jaw tips (clearance)
    - grasp width / angle / center from candidate
    """

    def __init__(self, config_path: Optional[str] = None, **kwargs):
        super().__init__(config_path=config_path, **kwargs)

        # Pixel-domain heuristic widths (camera/calibration dependent).
        # Keep this configurable per site if needed.
        self.config["gripper"]["min_width"] = 30.0
        self.config["gripper"]["max_width"] = 80.0
        self.config["gripper"]["ideal_width"] = 55.0

        w = self.config["evaluation"]["rule_weights"]
        w.clear()
        w["width"] = 0.22
        w["score"] = 0.10
        w["height"] = 0.08
        w["top_bias"] = 0.15
        w["neck_capture"] = 0.20
        w["support"] = 0.15
        w["clearance"] = 0.10

    def _top_bias(self, g: GraspCandidate) -> float:
        if getattr(g, "mask", None) is None:
            return 0.5
        ys, _ = np.where(g.mask > 0)
        if len(ys) == 0:
            return 0.5
        y_min, y_max = np.min(ys), np.max(ys)
        h = y_max - y_min
        if h <= 0:
            return 0.5
        rel_y = (g.center_y - y_min) / h
        # Favor upper-middle area (common for hook/neck capture).
        return float(np.exp(-0.5 * ((rel_y - 0.30) / 0.22) ** 2))

    def _neck_capture_fitness(self, g: GraspCandidate) -> float:
        if getattr(g, "mask", None) is None:
            return 0.5

        ys, _ = np.where(g.mask > 0)
        if len(ys) == 0:
            return 0.5
        y_min, y_max = np.min(ys), np.max(ys)
        neck_threshold = y_min + (y_max - y_min) * 0.38

        cx, cy = g.center_x, g.center_y
        r = max(1.0, g.width / 2.0)
        h, w = g.mask.shape

        top_contacts = 0
        bottom_contacts = 0
        for i in range(self.jaw_count):
            theta = g.angle + i * (2.0 * math.pi / float(self.jaw_count))
            jx = int(round(cx + r * math.cos(theta)))
            jy = int(round(cy + r * math.sin(theta)))
            if 0 <= jx < w and 0 <= jy < h and g.mask[jy, jx] > 0:
                if jy <= neck_threshold:
                    top_contacts += 1
                else:
                    bottom_contacts += 1

        if top_contacts >= 2 and bottom_contacts == 0:
            return 1.0
        if top_contacts == 1 and bottom_contacts == 0:
            return 0.75
        if bottom_contacts > 0:
            return 0.35
        return 0.55

    def _support_fitness(self, g: GraspCandidate) -> float:
        if getattr(g, "mask", None) is None:
            return 0.5

        h, w = g.mask.shape
        cx = int(_clip(g.center_x, 0, w - 1))
        cy = int(_clip(g.center_y, 0, h - 1))

        check_h = min(40, h - cy)
        if check_h <= 3:
            return 0.2

        x1 = max(0, cx - 20)
        x2 = min(w, cx + 21)
        region = g.mask[cy : cy + check_h, x1:x2]
        support_ratio = float(np.mean(region > 0))
        return float(_clip(support_ratio * 2.0, 0.0, 1.0))

    def _clearance_fitness(
        self, g: GraspCandidate, depth: Optional[np.ndarray]
    ) -> float:
        """
        Compare tip-depth and center-depth. If tip region appears deeper than center,
        it implies lower immediate collision risk around jaws.
        """
        if depth is None:
            return 0.5

        h, w = depth.shape
        cx = int(round(_clip(g.center_x, 0, w - 1)))
        cy = int(round(_clip(g.center_y, 0, h - 1)))
        center_d = float(depth[cy, cx])
        if not np.isfinite(center_d):
            return 0.5

        r = max(2.0, g.width / 2.0)
        tip_depths = []
        for i in range(self.jaw_count):
            theta = g.angle + i * (2.0 * math.pi / float(self.jaw_count))
            tx = int(round(cx + r * math.cos(theta)))
            ty = int(round(cy + r * math.sin(theta)))
            if 0 <= tx < w and 0 <= ty < h:
                d = float(depth[ty, tx])
                if np.isfinite(d):
                    tip_depths.append(d)

        if not tip_depths:
            return 0.5

        # Positive diff means tips are farther than center.
        diff = float(np.mean(tip_depths) - center_d)
        # 0cm -> 0.5, +2cm -> near 1.0, -2cm -> near 0.0
        return float(_clip(0.5 + diff / 0.04, 0.0, 1.0))

    def _width_fitness(self, g: GraspCandidate) -> float:
        spec = self.config["gripper"]
        if g.width < spec["min_width"] or g.width > spec["max_width"]:
            return 0.0
        ideal = spec["ideal_width"]
        if g.width <= ideal:
            return float(
                0.65
                + 0.35
                * (g.width - spec["min_width"])
                / (ideal - spec["min_width"] + 1e-6)
            )
        sigma = (spec["max_width"] - ideal) / 3.0
        return float(np.exp(-0.5 * ((g.width - ideal) / sigma) ** 2))

    def _calculate_rule_score(
        self, g: GraspCandidate, depth: Optional[np.ndarray]
    ) -> float:
        w = self.config["evaluation"]["rule_weights"]
        score = (
            w.get("width", 0.0) * self._width_fitness(g)
            + w.get("score", 0.0) * self._original_score(g)
            + w.get("height", 0.0) * self._height_fitness(g)
            + w.get("top_bias", 0.0) * self._top_bias(g)
            + w.get("neck_capture", 0.0) * self._neck_capture_fitness(g)
            + w.get("support", 0.0) * self._support_fitness(g)
            + w.get("clearance", 0.0) * self._clearance_fitness(g, depth)
        )
        # Keep score as ranking signal, not absolute probability.
        return float(_clip(score, 0.0, 0.98))

    def score_detail(self, g: GraspCandidate, depth: Optional[np.ndarray] = None) -> dict:
        w = self.config["evaluation"]["rule_weights"]

        rw = self._width_fitness(g)
        rs = self._original_score(g)
        rh = self._height_fitness(g)
        rt = self._top_bias(g)
        rn = self._neck_capture_fitness(g)
        ru = self._support_fitness(g)
        rc = self._clearance_fitness(g, depth)

        total = (
            w.get("width", 0.0) * rw
            + w.get("score", 0.0) * rs
            + w.get("height", 0.0) * rh
            + w.get("top_bias", 0.0) * rt
            + w.get("neck_capture", 0.0) * rn
            + w.get("support", 0.0) * ru
            + w.get("clearance", 0.0) * rc
        )

        total = float(_clip(total, 0.0, 0.98))
        return {
            "width": {"raw": rw, "weighted": w.get("width", 0.0) * rw},
            "score": {"raw": rs, "weighted": w.get("score", 0.0) * rs},
            "height": {"raw": rh, "weighted": w.get("height", 0.0) * rh},
            "top_bias": {"raw": rt, "weighted": w.get("top_bias", 0.0) * rt},
            "neck_capture": {
                "raw": rn,
                "weighted": w.get("neck_capture", 0.0) * rn,
            },
            "support": {"raw": ru, "weighted": w.get("support", 0.0) * ru},
            "clearance": {"raw": rc, "weighted": w.get("clearance", 0.0) * rc},
            # Backward-compatible aliases for old dashboards/log prints.
            "hooking": {"raw": rn, "weighted": w.get("neck_capture", 0.0) * rn},
            "slip_risk": {"raw": ru, "weighted": w.get("support", 0.0) * ru},
            "grip_shape": {"raw": rc, "weighted": w.get("clearance", 0.0) * rc},
            "total": total,
        }
