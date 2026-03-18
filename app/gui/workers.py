# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:45:31 2026

@author: michal.kalapus
"""

from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from app.services.workflows import ProjectionJob, process_projection_job
from ringremoval.engine import Params


class WorkerSignals(QObject):
    log = Signal(str)
    progress = Signal(int, int)
    finished = Signal(dict)
    error = Signal(str)


class ProjectionJobWorker(QRunnable):
    def __init__(self, job: ProjectionJob, params: Params) -> None:
        super().__init__()
        self.job = job
        self.params = params
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = process_projection_job(
                job=self.job,
                params=self.params,
                log=self.signals.log.emit,
                progress=self.signals.progress.emit,
            )
            self.signals.finished.emit(result)
        except Exception:
            self.signals.error.emit(traceback.format_exc())