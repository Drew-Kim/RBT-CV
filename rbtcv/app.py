from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import subprocess
import threading
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .annotations import AnnotationStore, TrialAnnotation, now_stamp
from .dataset import DatasetIndex, ROOT, TrialVideo
from .detection import (
    DISPLAY_LIKELIHOOD_CUTOFF,
    SCORING_LIKELIHOOD_CUTOFF,
    DLCPredictionStore,
    DLCTracking,
    draw_tracking_overlay,
)
from .scoring import (
    BEAM_LENGTH_CM,
    BEAM_TICK_MARKS_CM,
    DISTANCE_MARKS_CM,
    OUTCOME_FELL,
    OUTCOME_REACHED,
    PawMark,
    distance_for_outcome,
    max_time_applied,
    normalize_distance_cm,
    normalize_outcome,
    raw_crossing_time_seconds,
    result_text,
    scored_crossing_time_seconds,
)

from .results_workbook import ResultsWorkbook, ResultsWorkbookError
from .tail_position import TailPositionStore
from .tracking_rules import FELL, REACHED, analyze_tracking_timeline

from .ticks import (
    BeamCalibration,
    TickCalibrationStore,
    DLCTickDetector,
    calibration_from_detection,
    calibration_key,
    calibration_with_replaced_tick,
    estimate_distance_from_point,
    point_for_distance,
)


class RBTReviewApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("RBT-CV Review")
        self.minsize(1020, 660)
        self.geometry("1366x768")
        try:
            self.state("zoomed")
        except tk.TclError:
            pass

        self.dataset: DatasetIndex | None = None
        self.annotation_store = AnnotationStore()
        self.saved_annotations = self.annotation_store.load_by_video()
        self.tracking_store = DLCPredictionStore(likelihood_cutoff=DISPLAY_LIKELIHOOD_CUTOFF)
        self.dlc_tick_detector = DLCTickDetector()
        self.results_workbook = ResultsWorkbook()
        self.tail_position_store = TailPositionStore()
        self.current_tracking: DLCTracking | None = None
        self.current_scoring_tracking: DLCTracking | None = None
        self.dlc_running = False
        self.tracking_video_path: Path | None = None
        self.tick_store = TickCalibrationStore()
        self.saved_tick_calibrations = self.tick_store.load_by_key()

        self.cap: cv2.VideoCapture | None = None
        self.current_video: TrialVideo | None = None
        self.current_frame = 0
        self.frame_count = 0
        self.fps = 15.0
        self.video_width = 640
        self.video_height = 480
        self.display_scale = 1.0
        self.display_width = 640
        self.display_height = 480
        self.display_resize_id: str | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.playing = False
        self.after_id: str | None = None
        self.active_tick_distance_cm: int | None = None
        self.marks: dict[str, PawMark] = {}
        self.current_calibration: BeamCalibration | None = None
        self.pending_calibration: BeamCalibration | None = None
        self.updating_slider = False

        self.day_var = tk.StringVar()
        self.trial_count_var = tk.IntVar(value=3)
        self.status_var = tk.StringVar(value="Ready")
        self.frame_var = tk.StringVar(value="Frame -")
        self.time_var = tk.StringVar(value="Time -")
        self.result_var = tk.StringVar(value="Crossing time -")
        self.detection_enabled_var = tk.BooleanVar(value=False)
        self.tick_overlay_var = tk.BooleanVar(value=True)
        self.tick_status_var = tk.StringVar(value="No tick calibration")
        self.tick_edit_distance_var = tk.IntVar(value=0)
        self.outcome_var = tk.StringVar(value=OUTCOME_REACHED)
        self.distance_var = tk.IntVar(value=BEAM_LENGTH_CM)

        self.subject_labels: list[tuple[str, str]] = []
        self.trial_buttons: dict[int, ttk.Button] = {}

        self._build_ui()
        self._bind_keys()
        self._reset_dataset_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._build_top_bar()

        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))
        self._build_sidebar(main)
        self._build_video_panel(main)

        status = ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(10, 4))
        status.grid(row=2, column=0, sticky="ew")

    def _build_top_bar(self) -> None:
        top = ttk.Frame(self, padding=(10, 8))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Dataset").grid(row=0, column=0, sticky="w")
        self.dataset_label = ttk.Label(top, text="No video folder selected")
        self.dataset_label.grid(row=0, column=1, sticky="w", padx=(8, 12))
        ttk.Button(top, text="Choose", command=self.choose_dataset).grid(row=0, column=2, padx=(0, 6))
        self.reload_button = ttk.Button(top, text="Reload", command=self.reload_dataset, state="disabled")
        self.reload_button.grid(row=0, column=3)
    def _build_sidebar(self, parent: ttk.PanedWindow) -> None:
        sidebar = ttk.Frame(parent, padding=(0, 0, 10, 0), width=250)
        parent.add(sidebar, weight=0)
        sidebar.columnconfigure(0, weight=1)

        self._build_day_controls(sidebar)
        self._build_subject_controls(sidebar)
        self._build_trial_buttons(sidebar)
        self._build_tick_controls(sidebar)
        self._build_detection_controls(sidebar)
        self._build_timing_controls(sidebar)
        self._build_day_batch_button(sidebar)
        self._build_result_box(sidebar)
    def _build_day_controls(self, sidebar: ttk.Frame) -> None:
        ttk.Label(sidebar, text="Day").grid(row=0, column=0, sticky="w")
        self.day_combo = ttk.Combobox(sidebar, textvariable=self.day_var, state="readonly", width=24)
        self.day_combo.grid(row=1, column=0, sticky="ew", pady=(2, 8))
        self.day_combo.bind("<<ComboboxSelected>>", lambda _event: self.populate_subjects())

        trial_count_row = ttk.Frame(sidebar)
        trial_count_row.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(trial_count_row, text="Trials per subject").pack(side=tk.LEFT)
        self.trial_spin = ttk.Spinbox(
            trial_count_row,
            from_=1,
            to=3,
            textvariable=self.trial_count_var,
            width=4,
            command=self.update_trial_buttons,
        )
        self.trial_spin.pack(side=tk.RIGHT)

    def _build_subject_controls(self, sidebar: ttk.Frame) -> None:
        ttk.Label(sidebar, text="Subjects").grid(row=3, column=0, sticky="w")
        list_frame = ttk.Frame(sidebar)
        list_frame.grid(row=4, column=0, sticky="ew", pady=(2, 8))
        list_frame.columnconfigure(0, weight=1)

        self.subject_list = tk.Listbox(list_frame, height=8, exportselection=False)
        self.subject_list.grid(row=0, column=0, sticky="ew")
        subject_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.subject_list.yview)
        subject_scroll.grid(row=0, column=1, sticky="ns")
        self.subject_list.configure(yscrollcommand=subject_scroll.set)
        self.subject_list.bind("<<ListboxSelect>>", self.on_subject_selected)
    def _build_trial_buttons(self, sidebar: ttk.Frame) -> None:
        trial_row = ttk.Frame(sidebar)
        trial_row.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        for trial in range(1, 4):
            button = ttk.Button(trial_row, text=f"T{trial}", command=lambda trial=trial: self.open_trial(trial))
            button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0 if trial == 1 else 4, 0))
            self.trial_buttons[trial] = button

    def on_subject_selected(self, _event: tk.Event | None = None) -> None:
        self.update_trial_buttons()
        self.open_trial(1)

    def _build_detection_controls(self, sidebar: ttk.Frame) -> None:
        box = ttk.LabelFrame(sidebar, text="DeepLabCut Tracking", padding=6)
        box.grid(row=7, column=0, sticky="ew", pady=(0, 8))
        box.columnconfigure(0, weight=1)

        self.detection_enabled_var.set(True)
        ttk.Checkbutton(
            box,
            text="Show live tracking (>= 0.20)",
            variable=self.detection_enabled_var,
            command=lambda: self.show_frame(self.current_frame),
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            box,
            text="Analyze Current Trial",
            command=self.analyze_current_tracking,
        ).grid(row=1, column=0, sticky="ew", pady=(5, 0))
        ttk.Button(
            box,
            text="Analyze Selected Animal (T1-T3)",
            command=self.analyze_selected_animal,
        ).grid(row=2, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(
            box,
            text="Analyze Selected Day",
            command=self.analyze_selected_day,
        ).grid(row=3, column=0, sticky="ew", pady=(4, 0))



    def _build_tick_controls(self, sidebar: ttk.Frame) -> None:
        tick_box = ttk.LabelFrame(sidebar, text="Distance Interval Calibration", padding=6)
        tick_box.grid(row=6, column=0, sticky="ew", pady=(0, 8))
        tick_box.columnconfigure(0, weight=1)
        tick_box.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            tick_box,
            text="Show tick marks",
            variable=self.tick_overlay_var,
            command=self.refresh_frame,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Button(tick_box, text="Detect Trial's Ticks", command=self.auto_detect_ticks).grid(
            row=1, column=0, sticky="ew", pady=(6, 0), padx=(0, 4)
        )
        ttk.Button(tick_box, text="Detect Day's Ticks", command=self.auto_detect_day_ticks).grid(
            row=1, column=1, sticky="ew", pady=(6, 0)
        )
        ttk.Button(tick_box, text="Confirm Intervals", command=self.confirm_tick_calibration).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0)
        )

        edit_row = ttk.Frame(tick_box)
        edit_row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(edit_row, text="Edit").pack(side=tk.LEFT)
        self.tick_edit_combo = ttk.Combobox(
            edit_row,
            values=[str(value) for value in BEAM_TICK_MARKS_CM],
            textvariable=self.tick_edit_distance_var,
            state="readonly",
            width=6,
        )
        self.tick_edit_combo.pack(side=tk.LEFT, padx=(8, 4))
        ttk.Label(edit_row, text="cm").pack(side=tk.LEFT)
        ttk.Button(edit_row, text="Set from click", command=self.begin_tick_edit).pack(side=tk.RIGHT)

        ttk.Label(tick_box, textvariable=self.tick_status_var, wraplength=220).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )

    def _build_timing_controls(self, sidebar: ttk.Frame) -> None:
        mark_box = ttk.LabelFrame(sidebar, text="Trial Outcome", padding=6)
        mark_box.grid(row=8, column=0, sticky="ew", pady=(0, 8))
        mark_box.columnconfigure(0, weight=1)
        mark_box.columnconfigure(1, weight=1)

        outcome_row = ttk.Frame(mark_box)
        outcome_row.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Radiobutton(
            outcome_row,
            text="Reached platform",
            variable=self.outcome_var,
            value=OUTCOME_REACHED,
            command=self.on_outcome_changed,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            outcome_row,
            text="Fell",
            variable=self.outcome_var,
            value=OUTCOME_FELL,
            command=self.on_outcome_changed,
        ).pack(side=tk.LEFT, padx=(8, 0))

        distance_row = ttk.Frame(mark_box)
        distance_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        ttk.Label(distance_row, text="Distance").pack(side=tk.LEFT)
        self.distance_combo = ttk.Combobox(
            distance_row,
            values=[str(value) for value in DISTANCE_MARKS_CM],
            textvariable=self.distance_var,
            state="disabled",
            width=6,
        )
        self.distance_combo.pack(side=tk.LEFT, padx=(8, 4))
        self.distance_combo.bind("<<ComboboxSelected>>", lambda _event: self.on_distance_changed())
        ttk.Label(distance_row, text="cm").pack(side=tk.LEFT)

        ttk.Button(mark_box, text="Save Trial Result", command=self.save_annotation).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(7, 0)
        )
    def _build_day_batch_button(self, sidebar: ttk.Frame) -> None:
        self.day_batch_button = ttk.Button(
            sidebar,
            text="Calibrate & Analyze Day",
            command=self.calibrate_and_analyze_day,
        )
        self.day_batch_button.grid(row=9, column=0, sticky="ew", pady=(0, 8))

    def _build_result_box(self, sidebar: ttk.Frame) -> None:
        result_box = ttk.LabelFrame(sidebar, text="Scoring Preview", padding=6)
        result_box.grid(row=10, column=0, sticky="ew")
        ttk.Label(result_box, textvariable=self.result_var).grid(row=0, column=0, sticky="w")
    def _build_video_panel(self, parent: ttk.PanedWindow) -> None:
        self.video_panel = ttk.Frame(parent)
        self.video_panel.columnconfigure(0, weight=1)
        self.video_panel.rowconfigure(0, weight=1)
        parent.add(self.video_panel, weight=1)

        self._build_video_canvas(self.video_panel)
        self._build_playback_controls(self.video_panel)
        self.video_panel.bind("<Configure>", self._on_video_panel_resize)
    def _build_video_canvas(self, video_panel: ttk.Frame) -> None:
        self.canvas = tk.Canvas(video_panel, width=640, height=480, bg="#000000", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="n")
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def _build_playback_controls(self, video_panel: ttk.Frame) -> None:
        controls = ttk.Frame(video_panel, padding=(0, 8, 0, 0))
        self.playback_controls = controls
        controls.grid(row=1, column=0, sticky="ew")
        controls.columnconfigure(4, weight=1)

        ttk.Button(controls, text="|<", width=4, command=lambda: self.step_frame(-15)).grid(row=0, column=0)
        ttk.Button(controls, text="<", width=4, command=lambda: self.step_frame(-1)).grid(row=0, column=1, padx=(4, 0))
        self.play_button = ttk.Button(controls, text="Play", width=8, command=self.toggle_play)
        self.play_button.grid(row=0, column=2, padx=4)
        ttk.Button(controls, text=">", width=4, command=lambda: self.step_frame(1)).grid(row=0, column=3)
        self.slider = ttk.Scale(controls, from_=0, to=0, orient=tk.HORIZONTAL, command=self.on_seek)
        self.slider.grid(row=0, column=4, sticky="ew", padx=8)
        ttk.Label(controls, textvariable=self.frame_var, width=16).grid(row=0, column=5)
        ttk.Label(controls, textvariable=self.time_var, width=14).grid(row=0, column=6)

    def _on_video_panel_resize(self, _event: tk.Event) -> None:
        self._schedule_display_fit()

    def _schedule_display_fit(self) -> None:
        if self.display_resize_id is None:
            self.display_resize_id = self.after_idle(self._fit_video_display)

    def _fit_video_display(self) -> None:
        self.display_resize_id = None
        panel_width = self.video_panel.winfo_width()
        panel_height = self.video_panel.winfo_height()
        controls_height = max(self.playback_controls.winfo_reqheight(), 36)
        available_width = panel_width - 12
        available_height = panel_height - controls_height - 12
        if available_width <= 1 or available_height <= 1:
            return

        scale = min(
            1.5,
            available_width / max(self.video_width, 1),
            available_height / max(self.video_height, 1),
        )
        # The previous native-size display is always available as a safe minimum.
        scale = max(1.0, scale)
        width = max(1, int(round(self.video_width * scale)))
        height = max(1, int(round(self.video_height * scale)))
        if width == self.display_width and height == self.display_height:
            return

        self.display_scale = width / max(self.video_width, 1)
        self.display_width = width
        self.display_height = height
        self.canvas.config(width=width, height=height)
        if self.cap is not None:
            self.show_frame(self.current_frame)

    def _canvas_point(self, x: float, y: float) -> tuple[float, float]:
        return x * self.display_scale, y * self.display_scale

    def _video_point(self, x: float, y: float) -> tuple[int, int]:
        scale = self.display_scale or 1.0
        video_x = int(round(x / scale))
        video_y = int(round(y / scale))
        return (
            max(0, min(video_x, self.video_width - 1)),
            max(0, min(video_y, self.video_height - 1)),
        )

    def _canvas_image(self, frame_rgb) -> Image.Image:
        image = Image.fromarray(frame_rgb)
        target_size = (self.display_width, self.display_height)
        if image.size == target_size:
            return image
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        return image.resize(target_size, resampling)
    def _bind_keys(self) -> None:
        self.bind("<space>", lambda _event: self.toggle_play())
        self.bind("<Left>", lambda _event: self.step_frame(-1))
        self.bind("<Right>", lambda _event: self.step_frame(1))

    def _clear_video_panel(self) -> None:
        self.pause()
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.current_video = None
        self.current_tracking = None
        self.current_scoring_tracking = None
        self.tracking_video_path = None
        self.current_frame = 0
        self.frame_count = 0
        self.fps = 15.0
        self.video_width = 640
        self.video_height = 480
        self.display_scale = 1.0
        self.display_width = 640
        self.display_height = 480
        self.photo = None
        self.marks = {}
        self.current_calibration = None
        self.pending_calibration = None
        self.active_tick_distance_cm = None
        self.canvas.delete("all")
        self.canvas.config(width=self.display_width, height=self.display_height, bg="#000000")
        self._schedule_display_fit()
        self.slider.configure(to=0)
        self.updating_slider = True
        self.slider.set(0)
        self.updating_slider = False
        self.frame_var.set("Frame -")
        self.time_var.set("Time -")
        self.result_var.set("Crossing time -")

    def _reset_dataset_ui(self) -> None:
        self._clear_video_panel()
        self.day_combo.configure(values=(), state="disabled")
        self.day_var.set("")
        self.trial_spin.configure(state="disabled")
        self.subject_list.configure(state="normal")
        self.subject_list.delete(0, tk.END)
        self.subject_list.configure(state="disabled")
        self.subject_labels = []
        for button in self.trial_buttons.values():
            button.configure(state="disabled")
        self.dataset_label.configure(text="No video folder selected")
        self.reload_button.configure(state="disabled")
        self.tick_status_var.set("Choose a video folder first.")
        self.status_var.set("Choose a video folder to begin.")

    def _populate_days(self) -> None:
        if self.dataset is None:
            self._reset_dataset_ui()
            return

        days = self.dataset.days
        self.day_combo.configure(state="readonly", values=days)
        self.trial_spin.configure(state="normal")
        self.subject_list.configure(state="normal")
        if days:
            self.day_var.set(days[0])
            self.populate_subjects()
        else:
            self.day_var.set("")
            self.populate_subjects()
            self.status_var.set("No compatible videos found in this folder.")

    def choose_dataset(self) -> None:
        selected = filedialog.askdirectory(initialdir=str(ROOT / "data"), title="Choose RBT dataset folder")
        if not selected:
            return

        try:
            index = DatasetIndex(Path(selected))
        except (OSError, ValueError) as exc:
            messagebox.showerror("Dataset error", f"Could not load this video folder:\n{exc}")
            return
        if not index.all_videos:
            messagebox.showerror("No videos found", "The selected folder does not contain compatible RBT AVI videos.")
            return

        self._clear_video_panel()
        self.dataset = index
        self.saved_annotations = self.annotation_store.load_by_video()
        self.saved_tick_calibrations = self.tick_store.load_by_key()
        self.dataset_label.config(text=index.label)
        self.reload_button.configure(state="normal")
        self._populate_days()
        self.status_var.set("Dataset loaded. Select a subject and trial to begin.")

    def reload_dataset(self) -> None:
        if self.dataset is None:
            self.status_var.set("Choose a video folder first.")
            return

        try:
            index = DatasetIndex(self.dataset.dataset_dir)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Dataset error", f"Could not reload this video folder:\n{exc}")
            return
        if not index.all_videos:
            messagebox.showerror("No videos found", "This folder no longer contains compatible RBT AVI videos.")
            return

        self._clear_video_panel()
        self.dataset = index
        self.saved_annotations = self.annotation_store.load_by_video()
        self.saved_tick_calibrations = self.tick_store.load_by_key()
        self.dataset_label.config(text=index.label)
        self.reload_button.configure(state="normal")
        self._populate_days()
        self.status_var.set("Dataset reloaded. Select a subject and trial to begin.")
    def load_scoring_fields(self, trial_video: TrialVideo) -> None:
        row = self.saved_annotations.get(trial_video.relative_path)
        outcome = OUTCOME_REACHED
        distance_cm = BEAM_LENGTH_CM
        if row:
            outcome = normalize_outcome(row.get("outcome", ""))
            distance_cm = normalize_distance_cm(row.get("distance_cm", distance_cm))

        self.outcome_var.set(outcome)
        self.set_distance_cm(distance_for_outcome(outcome, distance_cm))
        self.update_distance_controls()

    def active_tick_calibration(self) -> BeamCalibration | None:
        return self.pending_calibration or self.current_calibration

    def calibration_for_video(self, trial_video: TrialVideo) -> BeamCalibration | None:
        """Use a T2/T3 recalibration when present; otherwise use the shared T1 calibration."""
        trial_key = calibration_key(trial_video, trial_specific=trial_video.trial != 1)
        return (
            self.saved_tick_calibrations.get(trial_key)
            or self.saved_tick_calibrations.get(calibration_key(trial_video))
        )

    def load_tick_calibration(self, trial_video: TrialVideo) -> None:
        self.active_tick_distance_cm = None
        self.pending_calibration = None
        self.current_calibration = self.calibration_for_video(trial_video)
        if self.current_calibration is not None:
            self.tick_status_var.set(
                f"Confirmed ticks loaded from T{self.current_calibration.source_trial} for this mouse/day."
            )
            return

        if trial_video.trial == 1:
            self.tick_status_var.set("No tick calibration. Detect or edit T1 ticks, then confirm intervals.")
        else:
            self.tick_status_var.set("No tick calibration. Detect this trial for a trial-specific recalibration, or use its T1 ticks.")


    def confirm_tick_calibration(self) -> None:
        if self.current_video is None:
            self.tick_status_var.set("Load T1 before confirming tick marks.")
            return
        calibration = self.active_tick_calibration()
        if calibration is None or not calibration.ticks:
            messagebox.showwarning("No ticks", "Auto-detect or set tick marks before confirming.")
            return

        confirmed = BeamCalibration(
            key=calibration_key(
                self.current_video,
                trial_specific=self.current_video.trial != 1,
            ),
            dataset=self.current_video.dataset,
            day=self.current_video.day,
            cage=self.current_video.cage_number,
            subject=self.current_video.rat_id,
            source_video=self.current_video.relative_path,
            source_trial=self.current_video.trial,
            frame_numbers=calibration.frame_numbers,
            ticks=calibration.ticks,
            confirmed_at=now_stamp(),
        )
        self.tick_store.save(confirmed)
        self.saved_tick_calibrations = self.tick_store.load_by_key()
        self.current_calibration = confirmed
        self.pending_calibration = None
        self.active_tick_distance_cm = None
        scope = "this mouse/day" if self.current_video.trial == 1 else f"this T{self.current_video.trial} trial"
        self.tick_status_var.set(f"Confirmed interval calibration saved for {scope}.")
        self.update_fall_distance_from_calibration()
        self.refresh_frame()

    def editable_calibration_for_current_video(self) -> BeamCalibration | None:
        if self.current_video is None:
            return None
        calibration = self.active_tick_calibration()
        if calibration is not None:
            return calibration
        return BeamCalibration(
            key=calibration_key(
                self.current_video,
                trial_specific=self.current_video.trial != 1,
            ),
            dataset=self.current_video.dataset,
            day=self.current_video.day,
            cage=self.current_video.cage_number,
            subject=self.current_video.rat_id,
            source_video=self.current_video.relative_path,
            source_trial=self.current_video.trial,
            frame_numbers=(self.current_frame,),
            ticks=(),
            confirmed_at="",
        )


    def begin_tick_edit(self) -> None:
        if self.current_video is None:
            self.tick_status_var.set("Load T1 before editing tick marks.")
            return
        calibration = self.editable_calibration_for_current_video()
        if calibration is None:
            return

        try:
            distance_cm = int(float(self.tick_edit_distance_var.get()))
        except (tk.TclError, ValueError):
            distance_cm = 0
        self.pending_calibration = calibration
        self.active_tick_distance_cm = distance_cm
        self.pause()
        self.tick_status_var.set(f"Click the {distance_cm} cm tick mark in the video.")

    def refresh_frame(self) -> None:
        if self.cap is not None:
            self.show_frame(self.current_frame)

    def on_outcome_changed(self) -> None:
        if self.outcome_var.get() == OUTCOME_REACHED:
            self.set_distance_cm(BEAM_LENGTH_CM)
        elif self.update_fall_distance_from_calibration():
            pass
        elif self.current_distance_cm() == BEAM_LENGTH_CM:
            self.set_distance_cm(0)
        else:
            self.set_distance_cm(self.current_distance_cm())

        self.update_distance_controls()
        self.update_result_labels()
        if self.cap is not None:
            self.show_frame(self.current_frame)

    def on_distance_changed(self) -> None:
        self.set_distance_cm(self.current_distance_cm())
        self.update_distance_controls()
        self.update_result_labels()
        if self.cap is not None:
            self.show_frame(self.current_frame)

    def set_distance_cm(self, distance_cm: int) -> None:
        self.distance_var.set(normalize_distance_cm(distance_cm))

    def current_distance_cm(self) -> int:
        try:
            selected_distance = int(float(self.distance_var.get()))
        except (tk.TclError, ValueError):
            selected_distance = 0
        return distance_for_outcome(self.outcome_var.get(), selected_distance)

    def update_distance_controls(self) -> None:
        state = "readonly" if self.outcome_var.get() == OUTCOME_FELL else "disabled"
        self.distance_combo.configure(state=state)



    def update_fall_distance_from_calibration(self) -> bool:
        if self.outcome_var.get() != OUTCOME_FELL:
            return False
        stop_mark = self.marks.get("stop")
        calibration = self.active_tick_calibration()
        if stop_mark is None or calibration is None:
            return False
        self.set_distance_cm(estimate_distance_from_point(calibration, stop_mark.x, stop_mark.y))
        return True

    def populate_subjects(self) -> None:
        self.subject_list.configure(state="normal")
        self.subject_list.delete(0, tk.END)
        self.subject_labels = []
        if self.dataset is None:
            self.update_trial_buttons()
            return

        self.subject_labels = self.dataset.subjects_for_day(self.day_var.get())
        for _key, label in self.subject_labels:
            self.subject_list.insert(tk.END, label)
        if self.subject_labels:
            self.subject_list.selection_set(0)
            self.subject_list.activate(0)
        self.update_trial_buttons()

    def selected_subject_key(self) -> str | None:
        if self.dataset is None:
            return None
        selection = self.subject_list.curselection()
        if not selection:
            return None
        label = self.subject_labels[selection[0]]
        return self.dataset.subject_key_from_label(label)

    def update_trial_buttons(self) -> None:
        subject_key = self.selected_subject_key()
        available = self.dataset.trials_for_subject(subject_key) if self.dataset is not None and subject_key else {}
        try:
            trial_count = int(self.trial_count_var.get())
        except tk.TclError:
            trial_count = 3

        for trial, button in self.trial_buttons.items():
            state = "normal" if trial <= trial_count and trial in available else "disabled"
            button.configure(state=state)

    def open_trial(self, trial: int) -> None:
        if self.dataset is None:
            self.status_var.set("Choose a video folder first.")
            return
        subject_key = self.selected_subject_key()
        if not subject_key:
            return
        trial_video = self.dataset.trials_for_subject(subject_key).get(trial)
        if trial_video is None:
            return
        self.load_video(trial_video)
    def load_video(self, trial_video: TrialVideo) -> None:
        self.pause()
        if self.cap is not None:
            self.cap.release()

        cap = cv2.VideoCapture(str(trial_video.path))
        if not cap.isOpened():
            messagebox.showerror("Video error", f"Could not open video:\n{trial_video.path}")
            return

        self.cap = cap
        self.current_video = trial_video
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        self.display_scale = 1.0
        self.display_width = self.video_width
        self.display_height = self.video_height
        self.canvas.config(width=self.display_width, height=self.display_height)
        self._schedule_display_fit()
        self.slider.configure(to=max(self.frame_count - 1, 0))
        self.marks = self._loaded_marks_for(trial_video)
        self.load_scoring_fields(trial_video)
        self.load_tick_calibration(trial_video)
        self.update_fall_distance_from_calibration()
        self.current_frame = 0
        self.show_frame(0)
        self.update_result_labels()
        self.status_var.set(f"Loaded {trial_video.relative_path}")

    def _loaded_marks_for(self, trial_video: TrialVideo) -> dict[str, PawMark]:
        row = self.saved_annotations.get(trial_video.relative_path)
        if not row:
            return {}
        try:
            return {
                "start": PawMark(int(row["start_frame"]), int(float(row["start_x"])), int(float(row["start_y"]))),
                "stop": PawMark(int(row["stop_frame"]), int(float(row["stop_x"])), int(float(row["stop_y"]))),
            }
        except (KeyError, ValueError):
            return {}



    def draw_tick_overlays(self) -> None:
        if not self.tick_overlay_var.get():
            return
        calibration = self.active_tick_calibration()
        if calibration is None:
            return

        pending = self.pending_calibration is not None
        color = "#ffe066" if pending else "#5cffc7"
        label_color = "#fff2a8" if pending else "#d6fff2"
        line_half = 14 * self.display_scale
        dot_radius = max(2, 3 * self.display_scale)
        line_width = max(1, int(round(2 * self.display_scale)))
        for tick in calibration.ticks:
            x, y = self._canvas_point(tick.x, tick.y)
            self.canvas.create_line(x, y - line_half, x, y + line_half, fill=color, width=line_width)
            self.canvas.create_oval(x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius, outline=color, width=line_width)
            self.canvas.create_text(
                x + 5 * self.display_scale,
                y - 18 * self.display_scale,
                text=f"{tick.distance_cm}",
                fill=label_color,
                anchor="w",
            )


    def frame_time(self, frame_number: int) -> float:
        return frame_number / self.fps if self.fps else 0.0

    def on_seek(self, value: str) -> None:
        if self.updating_slider:
            return
        if self.cap is None:
            return
        self.show_frame(int(float(value)))

    def step_frame(self, delta: int) -> None:
        if self.cap is None:
            return
        self.pause()
        self.show_frame(self.current_frame + delta)

    def toggle_play(self) -> None:
        if self.cap is None:
            return
        if self.playing:
            self.pause()
        else:
            self.playing = True
            self.play_button.configure(text="Pause")
            self.schedule_next_frame()

    def pause(self) -> None:
        self.playing = False
        self.play_button.configure(text="Play")
        if self.after_id is not None:
            self.after_cancel(self.after_id)
            self.after_id = None

    def schedule_next_frame(self) -> None:
        if not self.playing:
            return
        next_frame = self.current_frame + 1
        if next_frame >= self.frame_count:
            self.pause()
            return
        self.show_frame(next_frame)
        delay_ms = max(1, int(1000 / (self.fps or 15.0)))
        self.after_id = self.after(delay_ms, self.schedule_next_frame)


    def on_canvas_click(self, event: tk.Event) -> None:
        if self.cap is None:
            return
        if self.active_tick_distance_cm is None:
            self.status_var.set("Use Set from click before selecting a tick position.")
            return
        self.place_tick_from_click(event)



    def place_tick_from_click(self, event: tk.Event) -> None:
        calibration = self.active_tick_calibration()
        if calibration is None or self.active_tick_distance_cm is None:
            self.active_tick_distance_cm = None
            return

        x, y = self._video_point(event.x, event.y)
        distance_cm = self.active_tick_distance_cm
        self.pending_calibration = calibration_with_replaced_tick(calibration, distance_cm, x, y)
        self.active_tick_distance_cm = None
        self.tick_status_var.set(f"Moved {distance_cm} cm tick to ({x},{y}). Confirm Intervals to save.")
        self.update_fall_distance_from_calibration()
        self.update_distance_controls()
        self.update_result_labels()
        self.show_frame(self.current_frame)



    def raw_crossing_time(self) -> float | None:
        return raw_crossing_time_seconds(self.marks.get("start"), self.marks.get("stop"), self.fps)

    def crossing_time(self) -> float | None:
        return scored_crossing_time_seconds(self.outcome_var.get(), self.raw_crossing_time())

    def update_result_labels(self) -> None:
        raw_crossing = self.raw_crossing_time()
        crossing = self.crossing_time()
        distance_cm = self.current_distance_cm()
        self.result_var.set(result_text(self.outcome_var.get(), raw_crossing, crossing, distance_cm))

    def save_annotation(self) -> None:
        if self.current_video is None:
            self.status_var.set("Load a trial before saving.")
            return

        start = self.marks.get("start")
        stop = self.marks.get("stop")
        if start is None or stop is None:
            messagebox.showwarning(
                "Missing automatic result",
                "Analyze the trial first so the start and terminal event are available.",
            )
            return

        raw_crossing = self.raw_crossing_time()
        crossing = self.crossing_time()
        if crossing is None or raw_crossing is None or raw_crossing < 0:
            messagebox.showwarning(
                "Invalid timing",
                "The terminal event must be after the 0 cm start event.",
            )
            return

        outcome = self.outcome_var.get()
        distance_cm = self.current_distance_cm()
        annotation = TrialAnnotation(
            relative_video=self.current_video.relative_path,
            dataset=self.current_video.dataset,
            day=self.current_video.day,
            group=self.current_video.group,
            subject=self.current_video.subject,
            trial=self.current_video.trial,
            fps=self.fps,
            start_frame=start.frame,
            start_time=self.frame_time(start.frame),
            start_x=start.x,
            start_y=start.y,
            stop_frame=stop.frame,
            stop_time=self.frame_time(stop.frame),
            stop_x=stop.x,
            stop_y=stop.y,
            crossing_time=crossing,
            outcome=outcome,
            distance_cm=distance_cm,
            max_time_applied=max_time_applied(outcome),
            saved_at=now_stamp(),
        )

        try:
            self.annotation_store.save(annotation)
            self.results_workbook.save(annotation)
        except (OSError, ResultsWorkbookError) as exc:
            self.status_var.set(f"Saved annotation, but Excel could not be written: {exc}")
            return

        tail_refreshed = False
        try:
            calibration = self.calibration_for_video(self.current_video)
            if calibration is not None and self.current_scoring_tracking is not None:
                self.tail_position_store.record_trial(
                    self.current_video,
                    self.current_scoring_tracking,
                    calibration,
                    start.frame,
                    stop.frame,
                    refresh_plot=False,
                )
                tail_refreshed = (
                    self.tail_position_store.refresh_day_plot(
                        self.current_video.dataset,
                        self.current_video.day,
                    )
                    is not None
                )
        except OSError as exc:
            self.status_var.set(
                f"Saved trial result, but the tail graph could not be updated: {exc}"
            )
            return

        self.saved_annotations = self.annotation_store.load_by_video()
        tail_message = " Tail graph refreshed." if tail_refreshed else ""
        self.status_var.set(
            f"Saved trial result: {crossing:.2f} sec, distance {distance_cm} cm."
            f"{tail_message}"
        )


    def destroy(self) -> None:
        self.pause()
        if self.cap is not None:
            self.cap.release()
        super().destroy()


    # --- Restored DLC workflow -------------------------------------------------
    # DeepLabCut inference, automatic scoring, and result persistence.
    def _runtime(self) -> tuple[Path, Path, Path, Path]:
        python = ROOT / ".venv-dlc" / "Scripts" / "python.exe"
        runner = ROOT / "scripts" / "dlc_tracking.py"
        tracking_config = (
            ROOT
            / "models"
            / "dlc_tracking"
            / "RBT_visible_front_back_tail-RBT_CV-2026-07-14"
            / "config.yaml"
        )
        tick_config = (
            ROOT
            / "models"
            / "dlc_tickmarks"
            / "RBT_tick_landmarks-RBT_CV-2026-06-07"
            / "config.yaml"
        )
        return python, runner, tracking_config, tick_config

    def _save_calibration(self, video: TrialVideo, detection) -> bool:
        if {tick.distance_cm for tick in detection.ticks} != set(BEAM_TICK_MARKS_CM):
            return False

        calibration = calibration_from_detection(
            video,
            detection,
            now_stamp(),
            trial_specific=video.trial != 1,
        )
        self.tick_store.save(calibration)
        self.saved_tick_calibrations = self.tick_store.load_by_key()
        return True

    def auto_detect_ticks(self) -> None:
        if self.current_video is None:
            self.tick_status_var.set("Open a trial before detecting ticks.")
            return

        _python, _runner, _tracking_config, tick_config = self._runtime()
        video = self.current_video
        self._run_dlc(
            [
                "tick-analyze",
                "--config",
                str(tick_config),
                "--video",
                str(video.path),
                "--early-frames",
                "10",
            ],
            lambda code, output: self._finish_tick(video, code, output),
            "Detecting Trial's Ticks...",
        )

    def _finish_tick(self, video: TrialVideo, code: int, output: str) -> None:
        if code:
            self.tick_status_var.set(
                output.splitlines()[-1] if output else "Tick model failed."
            )
            return

        detection = self.dlc_tick_detector.detect_for_video(video)
        if not self._save_calibration(video, detection):
            self.tick_status_var.set(detection.message)
            return

        self.load_tick_calibration(video)
        self.tick_status_var.set(
            f"Tick Calibration Complete. {len(detection.ticks)}/13 ticks found."
        )
        if video == self.current_video:
            self.refresh_frame()

    def _automatic_annotation(
        self,
        video: TrialVideo,
    ) -> tuple[TrialAnnotation, DLCTracking, BeamCalibration] | None:
        calibration = self.calibration_for_video(video)
        csv_path = self.tracking_store.find_for_video(video)
        if calibration is None or csv_path is None:
            return None

        try:
            tracking = self.tracking_store.load(csv_path).filtered(
                SCORING_LIKELIHOOD_CUTOFF
            )
            timeline = analyze_tracking_timeline(tracking, calibration)
        except (OSError, ValueError):
            return None

        if (
            timeline.final_state not in {FELL, REACHED}
            or timeline.start_frame is None
            or timeline.end_frame is None
        ):
            return None

        outcome = OUTCOME_FELL if timeline.final_state == FELL else OUTCOME_REACHED
        distance_cm = (
            timeline.farthest_distance_cm
            if outcome == OUTCOME_FELL
            else BEAM_LENGTH_CM
        )
        start_tick = point_for_distance(calibration, 0)
        end_tick = point_for_distance(calibration, distance_cm)
        if start_tick is None or end_tick is None:
            return None

        stop_x, stop_y = end_tick.x, end_tick.y
        if outcome == OUTCOME_FELL:
            fall_state = timeline.state_at(timeline.end_frame)
            if (
                fall_state is not None
                and fall_state.body_center_x is not None
                and fall_state.body_center_y is not None
            ):
                stop_x = int(round(fall_state.body_center_x))
                stop_y = int(round(fall_state.body_center_y))

        capture = cv2.VideoCapture(str(video.path))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 15.0)
        finally:
            capture.release()

        start = PawMark(timeline.start_frame, start_tick.x, start_tick.y)
        stop = PawMark(timeline.end_frame, stop_x, stop_y)
        raw_time = raw_crossing_time_seconds(start, stop, fps)
        return (
            TrialAnnotation(
                relative_video=video.relative_path,
                dataset=video.dataset,
                day=video.day,
                group=video.group,
                subject=video.subject,
                trial=video.trial,
                fps=fps,
                start_frame=start.frame,
                start_time=start.frame / fps,
                start_x=start.x,
                start_y=start.y,
                stop_frame=stop.frame,
                stop_time=stop.frame / fps,
                stop_x=stop.x,
                stop_y=stop.y,
                crossing_time=scored_crossing_time_seconds(outcome, raw_time),
                outcome=outcome,
                distance_cm=distance_cm,
                max_time_applied=max_time_applied(outcome),
                saved_at=now_stamp(),
            ),
            tracking,
            calibration,
        )

    def _run_dlc(
        self,
        args: list[str],
        done: Callable[[int, str], None],
        message: str,
    ) -> None:
        if self.dlc_running:
            self.status_var.set("A DLC task is already running.")
            return

        python, runner, _tracking_config, _tick_config = self._runtime()
        if not python.exists() or not runner.exists():
            self.status_var.set("DeepLabCut environment is missing.")
            return

        self.dlc_running = True
        self.status_var.set(message)

        def work() -> None:
            try:
                process = subprocess.Popen(
                    [str(python), str(runner), *args],
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                output_lines: list[str] = []
                for line in process.stdout or ():
                    output_lines.append(line)
                    if line.startswith("RBT_PROGRESS	"):
                        parts = line.rstrip().split("	", 3)
                        if len(parts) == 4:
                            self.after(
                                0,
                                lambda parts=parts: self.status_var.set(
                                    f"Analyzing video {parts[1]}/{parts[2]}: "
                                    f"{Path(parts[3]).stem}"
                                ),
                            )
                code = process.wait()
                output = "".join(output_lines)
            except OSError as exc:
                code, output = 1, str(exc)

            def finish() -> None:
                self.dlc_running = False
                done(code, output)

            self.after(0, finish)

        threading.Thread(target=work, daemon=True).start()

    def _calibrate_day_ticks(
        self,
        on_complete: Callable[[int, int], None] | None = None,
    ) -> bool:
        if self.dataset is None or not self.day_var.get():
            self.tick_status_var.set("Choose a video folder first.")
            return False

        day = self.day_var.get()
        t1_videos = [
            video
            for video in self.dataset.videos
            if video.day == day and video.trial == 1
        ]
        if not t1_videos:
            self.tick_status_var.set("No T1 videos found for this day.")
            return False

        def run(index: int = 0, complete: int = 0) -> None:
            if index >= len(t1_videos):
                self.saved_tick_calibrations = self.tick_store.load_by_key()
                self.tick_status_var.set(
                    f"Tick Calibration Complete. {complete}/{len(t1_videos)} "
                    "T1 calibration(s) confirmed."
                )
                if on_complete is not None:
                    on_complete(complete, len(t1_videos))
                return

            video = t1_videos[index]
            _python, _runner, _tracking_config, tick_config = self._runtime()

            def finish(code: int, _output: str) -> None:
                succeeded = (
                    not code
                    and self._save_calibration(
                        video,
                        self.dlc_tick_detector.detect_for_video(video),
                    )
                )
                run(index + 1, complete + int(succeeded))

            self._run_dlc(
                [
                    "tick-analyze",
                    "--config",
                    str(tick_config),
                    "--video",
                    str(video.path),
                    "--early-frames",
                    "10",
                ],
                finish,
                (
                    f"Detecting day ticks {index + 1}/{len(t1_videos)}: "
                    f"Cage {video.cage_number} Rat {video.rat_id}, T1..."
                ),
            )

        run()
        return True

    def auto_detect_day_ticks(self) -> None:
        self._calibrate_day_ticks()

    def calibrate_and_analyze_day(self) -> None:
        if self.dataset is None or not self.day_var.get():
            self.status_var.set("Choose a video folder first.")
            return

        day = self.day_var.get()
        day_videos = [video for video in self.dataset.videos if video.day == day]
        if not day_videos:
            self.status_var.set("No videos found for this day.")
            return

        def begin_analysis(complete: int, total: int) -> None:
            if complete != total:
                self.status_var.set(
                    f"Day calibration incomplete ({complete}/{total}). "
                    "Day analysis was not started."
                )
                return
            self.status_var.set("Day ticks confirmed. Starting day analysis...")
            self._analyze(day_videos, f"{day} batch", refresh_tail=True)

        self.status_var.set(f"Calibrating {day} ticks before day analysis...")
        self._calibrate_day_ticks(begin_analysis)

    def _load_tracking(self, video: TrialVideo) -> bool:
        csv_path = self.tracking_store.find_for_video(video)
        self.tracking_video_path = video.path
        if csv_path is None:
            self.current_tracking = None
            self.current_scoring_tracking = None
            return False

        try:
            self.current_tracking = self.tracking_store.load(csv_path)
            self.current_scoring_tracking = self.current_tracking.filtered(
                SCORING_LIKELIHOOD_CUTOFF
            )
        except (OSError, ValueError):
            self.current_tracking = None
            self.current_scoring_tracking = None
            return False
        return True

    def _apply_automatic_current_result(self) -> None:
        if self.current_video is None:
            return

        result = self._automatic_annotation(self.current_video)
        if result is None:
            return

        annotation, _tracking, _calibration = result
        self.marks = {
            "start": PawMark(
                annotation.start_frame,
                annotation.start_x,
                annotation.start_y,
            ),
            "stop": PawMark(
                annotation.stop_frame,
                annotation.stop_x,
                annotation.stop_y,
            ),
        }
        self.outcome_var.set(annotation.outcome)
        self.set_distance_cm(annotation.distance_cm)
        self.update_distance_controls()
        self.update_result_labels()

    def draw_mark_overlays(self) -> None:
        for kind, mark in self.marks.items():
            if mark.frame > self.current_frame:
                continue

            is_fall = kind == "stop" and self.outcome_var.get() == OUTCOME_FELL
            x, y = mark.x, mark.y
            if is_fall and self.current_scoring_tracking is not None:
                prediction = self.current_scoring_tracking.points_for_frame(mark.frame)
                if prediction is not None and prediction.body_center is not None:
                    x = int(round(prediction.body_center.x))
                    y = int(round(prediction.body_center.y))

            display_x, display_y = self._canvas_point(x, y)
            color = "#ff4d4d" if is_fall else "#36e07a"
            label = "fall" if is_fall else (
                "0 cm start" if kind == "start" else "120 cm end"
            )
            radius = max(8, 10 * self.display_scale)
            line_width = max(1, int(round(3 * self.display_scale)))
            self.canvas.create_oval(
                display_x - radius,
                display_y - radius,
                display_x + radius,
                display_y + radius,
                outline=color,
                width=line_width,
            )
            self.canvas.create_text(
                display_x + 13 * self.display_scale,
                display_y - 13 * self.display_scale,
                text=label,
                fill=color,
                anchor="w",
            )

    def show_frame(self, frame_number: int) -> None:
        if self.cap is None:
            return
        if (
            self.current_video is not None
            and self.tracking_video_path != self.current_video.path
        ):
            self._load_tracking(self.current_video)

        frame_number = max(0, min(frame_number, max(self.frame_count - 1, 0)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = self.cap.read()
        if not ok:
            self.pause()
            return

        self.current_frame = frame_number
        prediction = (
            self.current_tracking.points_for_frame(frame_number)
            if self.current_tracking is not None
            else None
        )
        display = (
            draw_tracking_overlay(frame, prediction)
            if self.detection_enabled_var.get()
            else frame
        )
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=self._canvas_image(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.draw_tick_overlays()
        self.draw_mark_overlays()

        self.updating_slider = True
        self.slider.set(frame_number)
        self.updating_slider = False
        self.frame_var.set(
            f"Frame {frame_number + 1}/{max(self.frame_count, 1)}"
        )
        self.time_var.set(f"{self.frame_time(frame_number):.2f} sec")

    def _save_auto_batch(
        self,
        videos: list[TrialVideo],
        refresh_tail: bool = False,
    ) -> tuple[int, int, bool]:
        records: list[tuple[TrialVideo, TrialAnnotation, DLCTracking, BeamCalibration]] = []
        skipped = 0
        self.last_batch_save_error = ""

        for video in videos:
            result = self._automatic_annotation(video)
            if result is None:
                skipped += 1
                continue
            annotation, tracking, calibration = result
            records.append((video, annotation, tracking, calibration))

        if not records:
            self.saved_annotations = self.annotation_store.load_by_video()
            return 0, skipped, not refresh_tail

        annotations = [annotation for _video, annotation, _tracking, _calibration in records]
        try:
            self.annotation_store.save_many(annotations)
            self.results_workbook.save_many(annotations)
        except (OSError, ResultsWorkbookError) as exc:
            self.last_batch_save_error = str(exc)
            self.saved_annotations = self.annotation_store.load_by_video()
            return 0, skipped + len(records), False

        try:
            plots = self.tail_position_store.record_trials(
                [
                    (
                        video,
                        tracking,
                        calibration,
                        annotation.start_frame,
                        annotation.stop_frame,
                    )
                    for video, annotation, tracking, calibration in records
                ],
                refresh_plots=refresh_tail,
            )
            graph_ok = (
                not refresh_tail
                or bool(plots) and all(path is not None for path in plots.values())
            )
        except OSError as exc:
            self.last_batch_save_error = f"Results saved, but tail graph could not be written: {exc}"
            graph_ok = False

        self.saved_annotations = self.annotation_store.load_by_video()
        return len(records), skipped, graph_ok

    def _analyze(
        self,
        videos: list[TrialVideo],
        label: str,
        refresh_tail: bool = False,
    ) -> None:
        if not videos:
            self.status_var.set("No videos are available for analysis.")
            return
        if any(self.calibration_for_video(video) is None for video in videos):
            self.status_var.set("Detect and confirm ticks first.")
            return

        _python, _runner, tracking_config, _tick_config = self._runtime()
        args = ["analyze-files", "--config", str(tracking_config)]
        for video in videos:
            args.extend(("--video", str(video.path)))

        self._run_dlc(
            args,
            lambda code, output: self._finish_analysis(
                videos,
                code,
                output,
                refresh_tail,
            ),
            f"Analyzing {label}...",
        )

    def _finish_analysis(
        self,
        videos: list[TrialVideo],
        code: int,
        output: str,
        refresh_tail: bool,
    ) -> None:
        if code:
            self.status_var.set(
                output.splitlines()[-1] if output else "Tracking failed."
            )
            return

        saved, skipped, graph_ok = self._save_auto_batch(videos, refresh_tail)
        if self.current_video in videos:
            self._load_tracking(self.current_video)
            self._apply_automatic_current_result()
            self.show_frame(self.current_frame)

        graph_message = ""
        if refresh_tail:
            graph_message = (
                " Tail graph refreshed."
                if graph_ok
                else " Tail graph needs review."
            )
        error_message = (
            f" {self.last_batch_save_error}"
            if skipped and self.last_batch_save_error
            else ""
        )
        self.status_var.set(
            f"Analysis complete. {saved} result(s) saved to Excel; "
            f"{skipped} need review.{error_message}{graph_message}"
        )

    def analyze_current_tracking(self) -> None:
        if self.current_video is None:
            self.status_var.set("Load a trial before analyzing.")
            return
        self._analyze([self.current_video], "current trial", refresh_tail=True)

    def analyze_selected_animal(self) -> None:
        if self.dataset is None:
            self.status_var.set("Choose a video folder first.")
            return

        subject_key = self.selected_subject_key()
        videos = (
            list(self.dataset.trials_for_subject(subject_key).values())
            if subject_key
            else []
        )
        self._analyze(videos, "selected animal", refresh_tail=True)

    def analyze_selected_day(self) -> None:
        if self.dataset is None or not self.day_var.get():
            self.status_var.set("Choose a video folder first.")
            return

        videos = [
            video for video in self.dataset.videos if video.day == self.day_var.get()
        ]
        self._analyze(videos, "selected day", refresh_tail=True)
def check_app() -> int:
    dataset = DatasetIndex()
    print(f"Dataset: {dataset.dataset_dir.relative_to(ROOT)}")
    print(f"All videos: {len(dataset.all_videos)}")
    print(f"Review videos: {len(dataset.videos)}")
    survivors = ", ".join(f"Cage {cage} Rat {rat_id}" for _dataset, cage, rat_id in sorted(dataset.survivor_keys))
    print(f"D30 survivor roster: {survivors or 'not found; showing all videos'}")
    print(f"Days: {', '.join(dataset.days)}")
    print("Manual workbook comparison: disabled")
    return 0






def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RBT-CV video review GUI")
    parser.add_argument(
        "--check",
        action="store_true",
        help="load the current dataset without opening the GUI",
    )
    args = parser.parse_args(argv)

    if args.check:
        return check_app()

    app = RBTReviewApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
