# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:46:11 2026

@author: michal.kalapus
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QCheckBox,
    QComboBox,
    QPlainTextEdit,
    QProgressBar,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.gui.workers import ProjectionJobWorker
from app.services.workflows import ProjectionJob
from ringremoval.engine import Params


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Micro-CT Ring Removal")
        self.resize(1000, 700)

        self.thread_pool = QThreadPool.globalInstance()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        root.addWidget(self._build_input_group())
        root.addWidget(self._build_output_group())
        root.addWidget(self._build_algorithm_group())
        root.addWidget(self._build_run_group())
        root.addWidget(self._build_log_group())

    def _build_input_group(self) -> QGroupBox:
        box = QGroupBox("Input projection folders")
        layout = QVBoxLayout(box)

        self.folder_list = QListWidget()

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add folder(s)")
        remove_btn = QPushButton("Remove selected")
        clear_btn = QPushButton("Clear")

        add_btn.clicked.connect(self.add_folders)
        remove_btn.clicked.connect(self.remove_selected_folder)
        clear_btn.clicked.connect(self.folder_list.clear)

        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addWidget(clear_btn)

        pattern_row = QHBoxLayout()
        pattern_row.addWidget(QLabel("Glob pattern:"))
        self.glob_edit = QLineEdit("tomo_*.tif")
        self.recursive_check = QCheckBox("Recursive")
        pattern_row.addWidget(self.glob_edit)
        pattern_row.addWidget(self.recursive_check)

        layout.addLayout(btn_row)
        layout.addLayout(pattern_row)
        layout.addWidget(self.folder_list)
        return box

    def _build_output_group(self) -> QGroupBox:
        box = QGroupBox("Output")
        layout = QGridLayout(box)

        self.output_mode_combo = QComboBox()
        self.output_mode_combo.addItem("Inside input folder", "inside")
        self.output_mode_combo.addItem("One level above input folder", "up")
        self.output_mode_combo.addItem("One level below / child folder", "down")
        self.output_mode_combo.addItem("Custom folder", "custom")

        self.folder_name_edit = QLineEdit("ring_corrected")
        self.custom_output_edit = QLineEdit()
        self.custom_output_btn = QPushButton("Browse...")
        self.overwrite_check = QCheckBox("Overwrite output files")
        self.keep_temp_check = QCheckBox("Keep temporary sinogram files")

        self.custom_output_btn.clicked.connect(self.pick_custom_output_dir)
        self.output_mode_combo.currentIndexChanged.connect(self._update_output_mode_state)

        layout.addWidget(QLabel("Save mode:"), 0, 0)
        layout.addWidget(self.output_mode_combo, 0, 1, 1, 2)

        layout.addWidget(QLabel("Created folder name:"), 1, 0)
        layout.addWidget(self.folder_name_edit, 1, 1, 1, 2)

        layout.addWidget(QLabel("Custom output:"), 2, 0)
        layout.addWidget(self.custom_output_edit, 2, 1)
        layout.addWidget(self.custom_output_btn, 2, 2)

        layout.addWidget(self.overwrite_check, 3, 0, 1, 2)
        layout.addWidget(self.keep_temp_check, 4, 0, 1, 2)

        self._update_output_mode_state()
        return box

    def _build_algorithm_group(self) -> QGroupBox:
        box = QGroupBox("Algorithm")
        layout = QGridLayout(box)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["auto", "intensity", "log"])

        self.correction_combo = QComboBox()
        self.correction_combo.addItems([
            "auto", "algotom", "repair", "filtering", "sorting", "wavelet_fft", "dead", "large"
        ])
        self.correction_combo.setCurrentText("algotom")

        self.snr_spin = QLineEdit("3.0")
        self.la_size_spin = QSpinBox()
        self.la_size_spin.setRange(1, 100000)
        self.la_size_spin.setValue(51)

        self.sm_size_spin = QSpinBox()
        self.sm_size_spin.setRange(1, 100000)
        self.sm_size_spin.setValue(21)

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 128)
        self.workers_spin.setValue(12)
        self.workers_spin.setToolTip("0 = automatic")

        layout.addWidget(QLabel("Mode:"), 0, 0)
        layout.addWidget(self.mode_combo, 0, 1)

        layout.addWidget(QLabel("Correction:"), 1, 0)
        layout.addWidget(self.correction_combo, 1, 1)

        layout.addWidget(QLabel("SNR:"), 2, 0)
        layout.addWidget(self.snr_spin, 2, 1)

        layout.addWidget(QLabel("la_size:"), 3, 0)
        layout.addWidget(self.la_size_spin, 3, 1)

        layout.addWidget(QLabel("sm_size:"), 4, 0)
        layout.addWidget(self.sm_size_spin, 4, 1)

        layout.addWidget(QLabel("Workers:"), 5, 0)
        layout.addWidget(self.workers_spin, 5, 1)


        return box

    def _build_run_group(self) -> QGroupBox:
        box = QGroupBox("Run")
        layout = QHBoxLayout(box)

        self.run_btn = QPushButton("Run selected jobs")
        self.run_btn.clicked.connect(self.run_jobs)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(3)
        self.progress_bar.setValue(0)

        layout.addWidget(self.run_btn)
        layout.addWidget(self.progress_bar)
        return box

    def _build_log_group(self) -> QGroupBox:
        box = QGroupBox("Log")
        layout = QVBoxLayout(box)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)

        layout.addWidget(self.log_edit)
        return box

    def add_folders(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select projection folder")
        if not folder:
            return
        existing = {self.folder_list.item(i).text() for i in range(self.folder_list.count())}
        if folder not in existing:
            self.folder_list.addItem(folder)

    def remove_selected_folder(self) -> None:
        row = self.folder_list.currentRow()
        if row >= 0:
            self.folder_list.takeItem(row)

    def pick_custom_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select custom output folder")
        if folder:
            self.custom_output_edit.setText(folder)

    def _update_output_mode_state(self) -> None:
        mode = self.output_mode_combo.currentData()
        is_custom = mode == "custom"
        self.custom_output_edit.setEnabled(is_custom)
        self.custom_output_btn.setEnabled(is_custom)

    def append_log(self, text: str) -> None:
        self.log_edit.appendPlainText(text)

    def build_params(self) -> Params:
        return Params(
            mode=self.mode_combo.currentText(),
            correction=self.correction_combo.currentText(),
            snr=float(self.snr_spin.text()),
            la_size=self.la_size_spin.value(),
            sm_size=self.sm_size_spin.value(),
        )

    def build_jobs(self) -> list[ProjectionJob]:
        jobs: list[ProjectionJob] = []
        for i in range(self.folder_list.count()):
            jobs.append(
                ProjectionJob(
                    input_dir=self.folder_list.item(i).text(),
                    output_mode=self.output_mode_combo.currentData(),
                    folder_name=self.folder_name_edit.text().strip() or "ring_corrected",
                    custom_output_dir=self.custom_output_edit.text().strip() or None,
                    glob_pattern=self.glob_edit.text().strip() or "tomo_*.tif",
                    recursive=self.recursive_check.isChecked(),
                    overwrite=self.overwrite_check.isChecked(),
                    keep_temp=self.keep_temp_check.isChecked(),
                    workers=self.workers_spin.value(),
                )
            )
        return jobs

    def run_jobs(self) -> None:
        jobs = self.build_jobs()
        if not jobs:
            QMessageBox.warning(self, "No input", "Please add at least one projection folder.")
            return

        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_edit.clear()

        self.pending_jobs = len(jobs)
        params = self.build_params()

        for job in jobs:
            self.append_log(f"Queued: {job.input_dir}")
            worker = ProjectionJobWorker(job=job, params=params)
            worker.signals.log.connect(self.append_log)
            worker.signals.progress.connect(self.progress_bar.setValue)
            worker.signals.finished.connect(self.on_job_finished)
            worker.signals.error.connect(self.on_job_error)
            self.thread_pool.start(worker)

    def on_job_finished(self, result: dict) -> None:
        out_dir = result.get("output_dir", "")
        self.append_log(f"Finished successfully: {out_dir}")
        self._job_done()

    def on_job_error(self, error_text: str) -> None:
        self.append_log("ERROR:")
        self.append_log(error_text)
        self._job_done()

    def _job_done(self) -> None:
        self.pending_jobs -= 1
        if self.pending_jobs <= 0:
            self.run_btn.setEnabled(True)
            QMessageBox.information(self, "Done", "All jobs finished.")