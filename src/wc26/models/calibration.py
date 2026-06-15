"""Isotonic regression calibration for 3-class football probability outputs.

Fits one IsotonicRegression per class (H, D, A) on holdout predictions,
then renormalises the calibrated probabilities to sum to 1.

Requires a real holdout set — never fitted on training data.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class IsotonicCalibrator:
    """Per-class isotonic calibrator for multiclass probabilities."""

    def __init__(self) -> None:
        self._calibrators: list | None = None
        self._fitted = False

    def fit(self, probs: np.ndarray, outcomes: list[str]) -> None:
        """Fit calibrators on (n, 3) predicted probs vs observed outcomes.

        Parameters
        ----------
        probs    : shape (n, 3), columns are [p_home, p_draw, p_away]
        outcomes : list of "H", "D", "A" strings (same order)
        """
        from sklearn.isotonic import IsotonicRegression

        if len(outcomes) < 10:
            logger.warning(
                f"Only {len(outcomes)} calibration samples — isotonic may overfit. "
                "Need ≥30 for reliable calibration."
            )

        y = np.array(
            [[1, 0, 0] if o == "H" else ([0, 1, 0] if o == "D" else [0, 0, 1])
             for o in outcomes],
            dtype=float,
        )
        self._calibrators = []
        for i in range(3):
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(probs[:, i], y[:, i])
            self._calibrators.append(cal)

        self._fitted = True
        logger.info(f"Isotonic calibrators fitted on {len(outcomes)} samples")

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply calibration to (n, 3) array; renormalise rows to sum to 1."""
        if not self._fitted or self._calibrators is None:
            return probs

        cal = np.column_stack(
            [self._calibrators[i].predict(probs[:, i]) for i in range(3)]
        ).clip(0)
        row_sums = cal.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return cal / row_sums

    def calibrate_single(
        self, p_home: float, p_draw: float, p_away: float
    ) -> tuple[float, float, float]:
        arr = self.calibrate(np.array([[p_home, p_draw, p_away]]))[0]
        return float(arr[0]), float(arr[1]), float(arr[2])

    @property
    def is_fitted(self) -> bool:
        return self._fitted
