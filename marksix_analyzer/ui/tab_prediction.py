"""Prediction tab: runs the AI-based predictor and displays results."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..models import Draw, Pick
from ..db import Database
from ..predict import predict_once
from ..strings import s
from .widgets import BallRow, DisclaimerLabel


class PredictionTab(QWidget):
    def __init__(self, db: Database, parent: QWidget | None = None):
        super().__init__(parent)
        self.db = db
        self._draws: list[Draw] = []
        self._last_combo: list[int] | None = None

        root = QVBoxLayout(self)
        root.addWidget(QLabel(f"<b>{s('tab_prediction') if 'tab_prediction' in globals() else 'Prediction'}</b>"))

        # Metrics row (AI predicted macros)
        self.metrics_row = QWidget()
        mr_layout = QHBoxLayout(self.metrics_row)
        self.metric_sum = QLabel("Target Sum: —")
        self.metric_odd_even = QLabel("Odd/Even: —")
        self.metric_big_small = QLabel("Big/Small: —")
        for w in (self.metric_sum, self.metric_odd_even, self.metric_big_small):
            w.setAlignment(Qt.AlignCenter)
            w.setStyleSheet("font-weight:600; padding:8px; border:1px solid #444; border-radius:8px;")
            mr_layout.addWidget(w)
        mr_layout.addStretch(1)
        root.addWidget(self.metrics_row)

        # result row (balls + actions)
        self.result_row = QWidget()
        rr_layout = QHBoxLayout(self.result_row)
        self.ball_row = BallRow([], diameter=44)
        rr_layout.addWidget(self.ball_row, 1)
        self.save_btn = QPushButton(s("save") if "save" in globals() else "Save")
        self.copy_btn = QPushButton(s("copy") if "copy" in globals() else "Copy")
        rr_layout.addWidget(self.save_btn)
        rr_layout.addWidget(self.copy_btn)
        root.addWidget(self.result_row)

        # predict button
        self.predict_btn = QPushButton(s("predict") if "predict" in globals() else "Run prediction")
        self.predict_btn.setObjectName("accent")
        self.predict_btn.clicked.connect(self._on_predict)
        root.addWidget(self.predict_btn)

        root.addWidget(DisclaimerLabel(s("sp_disclaimer")))

        self.save_btn.clicked.connect(self._on_save)
        self.copy_btn.clicked.connect(self._on_copy)

    def set_data(self, draws: list[Draw], params: dict) -> None:
        # keep latest seed set from main window, but pull fresh data when predicting
        self._draws = draws

    def _on_predict(self) -> None:
        # Ensure UI remains responsive and multiple clicks are allowed
        self.predict_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Use fresh draws from DB to ensure latest state
            draws = self.db.all_draws()
            if not draws:
                QMessageBox.warning(self, s("info"), s("status_no_data"))
                return

            # Run prediction (may be somewhat slow); wrap exceptions
            combo = predict_once(draws, method="ai_feature_xgboost")
            if not combo:
                QMessageBox.warning(self, s("info"), s("sp_gen_failed") if "sp_gen_failed" in globals() else "Prediction failed")
                return

            # Update balls and store current combo
            self.ball_row.set_numbers(combo)
            self._last_combo = combo

            # Compute and show macro-features (sum, odd/even, big/small)
            total = sum(combo)
            odd = sum(1 for n in combo if n % 2 == 1)
            big = sum(1 for n in combo if 25 <= n <= 49)
            sum_tol = 15
            self.metric_sum.setText(f"Target Sum Range: {total} ± {sum_tol}")
            self.metric_odd_even.setText(f"Odd/Even: {odd} Odds / {6-odd} Evens")
            self.metric_big_small.setText(f"Big/Small: {big} Big / {6-big} Small")

            # Force repaint of ball row
            self.ball_row.update()

        except Exception as e:
            QMessageBox.warning(self, s("error") if "error" in globals() else "Error", str(e))
        finally:
            QApplication.restoreOverrideCursor()
            self.predict_btn.setEnabled(True)

    def _on_save(self) -> None:
        # Save current displayed pick
        nums = getattr(self, "_last_combo", None)
        if not nums:
            QMessageBox.information(self, s("info"), s("no_selection"))
            return
        self.db.save_pick(Pick(numbers=nums, method="ai_feature_xgboost"))
        QMessageBox.information(self, s("info"), s("sp_saved_ok"))

    def _on_copy(self) -> None:
        from PySide6.QtWidgets import QApplication

        nums = getattr(self, "_last_combo", None)
        if not nums:
            QMessageBox.information(self, s("info"), s("no_selection"))
            return
        QApplication.clipboard().setText(", ".join(str(n) for n in nums))
        QMessageBox.information(self, s("info"), s("sp_copied"))
