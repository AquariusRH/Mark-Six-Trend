"""QMainWindow: tabs, global filter bar, toolbar, status bar, background fetch."""
from __future__ import annotations

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import analysis
from ..config import (
    APP_AUTHOR,
    APP_NAME,
    APP_VERSION,
    SEED_CSV_PATH,
    SEED_VERSION,
)
from ..data_provider import (
    HKJCProvider,
    export_csv,
    import_csv,
    load_seed,
)
from ..db import Database
from ..models import Draw
from ..strings import s
from ..workers import FetchController
from .tab_dashboard import DashboardTab
from .tab_data import DataTab, ManualEntryDialog
from .tab_distribution import DistributionTab
from .tab_frequency import FrequencyTab
from .tab_gaps import GapsTab
from .tab_saved import SavedTab
from .tab_smartpick import SmartPickTab
from .tab_trend import TrendTab
from .tab_prediction import PredictionTab
from ..strings import get_lang, lang_button_label, other_lang, set_lang
from .widgets import BrandHeader, FilterBar


class MainWindow(QMainWindow):
    def __init__(self, db: Database):
        super().__init__()
        self.db = db
        self.provider = HKJCProvider()
        self._fetch_controller: FetchController | None = None
        self._toolbar = None

        self.setMinimumSize(1100, 720)
        self.settings = QSettings(APP_AUTHOR, APP_NAME)
        set_lang(self.settings.value("language", "zh"))
        self.setWindowTitle(s("app_title"))

        self._seed_if_empty()
        self._build_ui()
        self._restore_geometry()
        self.refresh_all()
        self._maybe_autofetch()

    # -- language -----------------------------------------------------------
    def _toggle_language(self) -> None:
        set_lang(other_lang())
        self.settings.setValue("language", get_lang())
        self._retranslate()

    def _retranslate(self) -> None:
        """Rebuild the UI in the newly selected language, preserving state."""
        # Remember view state so the switch feels seamless.
        tab_idx = self.tabs.currentIndex()
        fb = self.filter_bar
        state = {
            "mode": fb.combo.currentIndex(),
            "from": fb.from_edit.date(),
            "to": fb.to_edit.date(),
            "extra": fb.include_extra.isChecked(),
        }

        if self._toolbar is not None:
            self.removeToolBar(self._toolbar)
            self._toolbar.deleteLater()
        self.menuBar().clear()
        self._build_ui()  # replaces the central widget + toolbar + menu

        self.setWindowTitle(s("app_title"))
        self.filter_bar.combo.setCurrentIndex(state["mode"])
        self.filter_bar.from_edit.setDate(state["from"])
        self.filter_bar.to_edit.setDate(state["to"])
        self.filter_bar.include_extra.setChecked(state["extra"])
        self.tabs.setCurrentIndex(tab_idx)
        self.refresh_all()

    # -- setup --------------------------------------------------------------
    def _seed_if_empty(self) -> None:
        """Import bundled seed on first launch, or when it has been updated.

        Uses a seed_version marker so an existing DB seeded from an older
        bundle picks up the newer history. Upsert is idempotent and never
        touches saved_picks.
        """
        needs_seed = (
            self.db.count_draws() == 0
            or self.db.get_meta("seed_version") != SEED_VERSION
        )
        if needs_seed:
            draws = load_seed(SEED_CSV_PATH)
            if draws:
                self.db.upsert_draws(draws)
                self.db.set_meta("seed_version", SEED_VERSION)

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(14, 12, 14, 8)
        layout.setSpacing(10)

        self.header = BrandHeader(
            s("app_title"),
            s("header_subtitle", version=APP_VERSION),
            lang_button_label(),
        )
        self.header.lang_button.clicked.connect(self._toggle_language)
        layout.addWidget(self.header)

        self.filter_bar = FilterBar()
        self.filter_bar.changed.connect(self.refresh_all)
        layout.addWidget(self.filter_bar)

        self.tabs = QTabWidget()
        self.dashboard = DashboardTab()
        self.frequency = FrequencyTab()
        self.gaps = GapsTab()
        self.distribution = DistributionTab()
        self.trend = TrendTab()
        self.prediction = PredictionTab(self.db)
        self.smartpick = SmartPickTab(self.db)
        self.saved = SavedTab(self.db)
        self.data = DataTab(self.db)

        self.smartpick.saved = self.saved.reload  # refresh saved list on save

        self.tabs.addTab(self.dashboard, s("tab_dashboard"))
        self.tabs.addTab(self.frequency, s("tab_frequency"))
        self.tabs.addTab(self.gaps, s("tab_gaps"))
        self.tabs.addTab(self.distribution, s("tab_distribution"))
        self.tabs.addTab(self.trend, s("tab_trend"))
        self.tabs.addTab(self.prediction, s("tab_prediction"))
        self.tabs.addTab(self.smartpick, s("tab_smartpick"))
        self.tabs.addTab(self.saved, s("tab_saved"))
        self.tabs.addTab(self.data, s("tab_data"))
        layout.addWidget(self.tabs, 1)

        self.setCentralWidget(central)

        # data tab signals
        self.data.request_refresh.connect(self.start_fetch)
        self.data.request_import.connect(self.import_csv_dialog)
        self.data.request_export.connect(self.export_csv_dialog)
        self.data.request_manual_add.connect(self.manual_add_dialog)

        self._build_menu()
        self._update_status()

    def _build_menu(self) -> None:
        toolbar = self.addToolBar("main")
        toolbar.setMovable(False)
        self._toolbar = toolbar

        def add_action(label: str, slot) -> QAction:
            act = QAction(label, self)
            act.triggered.connect(slot)
            toolbar.addAction(act)
            return act

        add_action(s("action_refresh"), self.start_fetch)
        add_action(s("action_import_csv"), self.import_csv_dialog)
        add_action(s("action_export_csv"), self.export_csv_dialog)
        add_action(s("action_manual_add"), self.manual_add_dialog)
        toolbar.addSeparator()
        add_action(s("action_about"), self.show_about)

        menu = self.menuBar()
        file_menu = menu.addMenu(s("menu_file"))
        file_menu.addAction(s("action_import_csv"), self.import_csv_dialog)
        file_menu.addAction(s("action_export_csv"), self.export_csv_dialog)
        file_menu.addAction(s("action_manual_add"), self.manual_add_dialog)
        file_menu.addSeparator()
        file_menu.addAction(s("action_exit"), self.close)
        help_menu = menu.addMenu(s("menu_help"))
        help_menu.addAction(s("action_about"), self.show_about)

    # -- data refresh -------------------------------------------------------
    def refresh_all(self) -> None:
        draws = self.db.all_draws()
        params = self.filter_bar.params()
        filtered = analysis.filter_draws(
            draws,
            last_n=params["last_n"],
            date_from=params["date_from"],
            date_to=params["date_to"],
        )
        for tab in (self.dashboard, self.frequency, self.gaps,
                    self.distribution, self.trend, self.prediction, self.smartpick,
                    self.saved, self.data):
            tab.set_data(filtered, params)
        self._update_status()

    def _update_status(self, override: str | None = None) -> None:
        if override:
            self.statusBar().showMessage(override)
            return
        latest = self.db.latest_draw()
        if latest:
            self.statusBar().showMessage(
                s("status_latest", draw_id=latest.draw_id, date=latest.draw_date)
            )
            self.header.set_pill(
                f"● {s('dash_latest_draw')}　{latest.draw_id}　{latest.draw_date}"
            )
        else:
            self.statusBar().showMessage(s("status_no_data"))
            self.header.set_pill(s("status_no_data"))

    # -- background fetch ---------------------------------------------------
    def _maybe_autofetch(self) -> None:
        from datetime import date, timedelta

        latest = self.db.latest_draw()
        if latest is None:
            return
        try:
            last = date.fromisoformat(latest.draw_date)
        except ValueError:
            return
        if date.today() > last + timedelta(days=1):
            self.start_fetch()

    def start_fetch(self) -> None:
        if self._fetch_controller is not None:
            return
        self._update_status(s("status_updating"))
        latest = self.db.latest_draw()
        since = latest.draw_id if latest else None
        self._fetch_controller = FetchController(self.provider, since, self)
        self._fetch_controller.finished.connect(self._on_fetched)
        self._fetch_controller.failed.connect(self._on_fetch_failed)
        self._fetch_controller.start()

    def _on_fetched(self, draws: list) -> None:
        self._fetch_controller = None
        if draws:
            self.db.upsert_draws(draws)
            self.refresh_all()
        else:
            self._update_status(s("status_up_to_date"))
            self._update_status()

    def _on_fetch_failed(self, msg: str) -> None:
        self._fetch_controller = None
        self._update_status(s("status_update_failed"))

    # -- CSV / manual -------------------------------------------------------
    def import_csv_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, s("csv_import_title"), "", s("csv_filter")
        )
        if not path:
            return
        draws, errors = import_csv(path)
        if draws:
            self.db.upsert_draws(draws)
            self.refresh_all()
        msg = s("csv_import_done", ok=len(draws), bad=len(errors))
        if errors:
            lines = "\n".join(f"  行 {e.line}：{e.reason}" for e in errors[:30])
            msg += "\n\n" + s("csv_import_errors", lines=lines)
        QMessageBox.information(self, s("csv_import_title"), msg)

    def export_csv_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, s("action_export_csv"), "marksix_export.csv", s("csv_filter")
        )
        if not path:
            return
        count = export_csv(path, self.db.all_draws())
        QMessageBox.information(
            self, s("action_export_csv"),
            s("csv_export_done", count=count, path=path)
        )

    def manual_add_dialog(self) -> None:
        dlg = ManualEntryDialog(self)
        if dlg.exec() and dlg.result_draw():
            self.db.upsert_draw(dlg.result_draw())
            self.db.conn.commit()
            self.refresh_all()

    # -- about --------------------------------------------------------------
    def show_about(self) -> None:
        text = (
            f"<h3>{s('app_title')}</h3>"
            f"<p>{s('about_version', version=APP_VERSION)}</p>"
            f"<p><b>{s('disclaimer_title')}</b><br>"
            f"{s('disclaimer_body').replace(chr(10), '<br>')}</p>"
            f"<p>{s('about_data_source')}</p>"
            f"<p><i>{s('about_seed_note')}</i></p>"
        )
        QMessageBox.about(self, s("about_title"), text)

    # -- geometry persistence ----------------------------------------------
    def _restore_geometry(self) -> None:
        geo = self.settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        state = self.settings.value("windowState")
        if state is not None:
            self.restoreState(state)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        super().closeEvent(event)
