from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
import subprocess
import threading
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .annotations import AnnotationStore, TrialAnnotation, now_stamp
from .condition_map import ConditionMapStore
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
from .research_angle import calculate_tail_angle, tail_angle_frame_records
from .tail_position import TailPositionStore
from .tracking_rules import FELL, REACHED, analyze_tracking_timeline

from .ticks import (
    BeamCalibration,
    TickCalibrationStore,
    TICK_CALIBRATION_DRAFTS_FILE,
    DLCTickDetector,
    calibration_from_detection,
    calibration_key,
    calibration_with_replaced_tick,
    estimate_distance_from_point,
    point_for_distance,
)


INITIAL_TICK_SCAN_FRAMES = 10
BROADER_TICK_SCAN_FRAMES = 30


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
        self.condition_map_store = ConditionMapStore()
        self.tail_position_store = TailPositionStore()
        self.current_tracking: DLCTracking | None = None
        self.current_scoring_tracking: DLCTracking | None = None
        self.dlc_running = False
        self.tracking_video_path: Path | None = None
        self.tick_store = TickCalibrationStore()
        self.saved_tick_calibrations = self.tick_store.load_by_key()
        self.tick_draft_store = TickCalibrationStore(TICK_CALIBRATION_DRAFTS_FILE)
        self.saved_tick_drafts = self.tick_draft_store.load_by_key()

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
        self.guided_tick_sequence = False
        self.marks: dict[str, PawMark] = {}
        self.terminal_event = "manual"
        self.current_calibration: BeamCalibration | None = None
        self.pending_calibration: BeamCalibration | None = None
        self.updating_slider = False
        self.pending_seek_frame: int | None = None

        self.day_var = tk.StringVar()
        self.trial_count_var = tk.IntVar(value=3)
        self.status_var = tk.StringVar(value="Ready")
        self.frame_var = tk.StringVar(value="Frame -")
        self.time_var = tk.StringVar(value="Time -")
        self.result_var = tk.StringVar(value="Crossing time -")
        self.detection_enabled_var = tk.BooleanVar(value=False)
        self.angle_overlay_var = tk.BooleanVar(value=True)
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
        sidebar_host = ttk.Frame(parent, width=270)
        parent.add(sidebar_host, weight=0)
        sidebar_host.columnconfigure(0, weight=1)
        sidebar_host.rowconfigure(0, weight=1)

        sidebar_canvas = tk.Canvas(sidebar_host, highlightthickness=0, borderwidth=0)
        sidebar_canvas.grid(row=0, column=0, sticky="nsew")
        sidebar_scroll = ttk.Scrollbar(sidebar_host, orient=tk.VERTICAL, command=sidebar_canvas.yview)
        sidebar_scroll.grid(row=0, column=1, sticky="ns")
        sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)

        sidebar = ttk.Frame(sidebar_canvas, padding=(0, 0, 10, 0), width=250)
        sidebar_window = sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw")

        def update_sidebar_scrollregion(_event: tk.Event | None = None) -> None:
            sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all"))

        def resize_sidebar_content(event: tk.Event) -> None:
            sidebar_canvas.itemconfigure(sidebar_window, width=event.width)

        def scroll_sidebar(event: tk.Event) -> str:
            # Keep the mouse wheel local to this panel; the subject list keeps
            # its own normal scroll behavior.
            hovered = self.winfo_containing(event.x_root, event.y_root)
            if hovered is None or not str(hovered).startswith(str(sidebar)):
                return ""
            if isinstance(hovered, tk.Listbox):
                return ""
            delta = int(-event.delta / 120) if event.delta else 0
            if delta:
                sidebar_canvas.yview_scroll(delta, "units")
                return "break"
            return ""

        sidebar.bind("<Configure>", update_sidebar_scrollregion)
        sidebar_canvas.bind("<Configure>", resize_sidebar_content)
        self.bind_all("<MouseWheel>", scroll_sidebar, add="+")
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
        ttk.Button(
            list_frame,
            text="Label SHAM / STROKE Groups",
            command=self.open_condition_manager,
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
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

    def open_condition_manager(self) -> None:
        """Open a multi-select editor for dataset-wide SHAM/STROKE labels."""
        if self.dataset is None:
            self.status_var.set("Choose a dataset before labeling subjects.")
            return

        subjects = {
            (video.dataset, video.cage_number, video.rat_id)
            for video in self.dataset.videos
        }
        if not subjects:
            self.status_var.set("No subjects are available to label.")
            return

        assignments = {
            dataset: self.condition_map_store.load(dataset)
            for dataset, _cage, _animal in subjects
        }
        window = tk.Toplevel(self)
        window.title("Label SHAM / STROKE Groups")
        window.geometry("620x520")
        window.minsize(520, 380)
        window.transient(self)

        outer = ttk.Frame(window, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            outer,
            text="Select one or more Cage/Rat subjects, then assign one condition. Labels apply to every day and trial.",
            wraplength=560,
        ).pack(anchor="w", pady=(0, 8))

        table_frame = ttk.Frame(outer)
        table_frame.pack(fill=tk.BOTH, expand=True)
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        tree = ttk.Treeview(
            table_frame,
            columns=("cage", "rat", "condition"),
            show="headings",
            selectmode="extended",
            height=18,
        )
        tree.heading("cage", text="Cage")
        tree.heading("rat", text="Rat")
        tree.heading("condition", text="Condition")
        tree.column("cage", width=130, anchor="center")
        tree.column("rat", width=130, anchor="center")
        tree.column("condition", width=180, anchor="center")
        tree.tag_configure("SHAM", foreground="#1976D2")
        tree.tag_configure("STROKE", foreground="#D32F2F")
        tree.tag_configure("UNASSIGNED", foreground="#555555")
        tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        tree.configure(yscrollcommand=scroll.set)

        item_subjects: dict[str, tuple[str, str, str]] = {}

        def refresh_table() -> None:
            for item in tree.get_children():
                tree.delete(item)
            for dataset, cage, animal in sorted(
                subjects,
                key=lambda item: (_numeric_id_sort(item[1]), _numeric_id_sort(item[2]), item[0].casefold()),
            ):
                condition = assignments[dataset].get((cage, animal), "Unassigned")
                item_id = tree.insert(
                    "",
                    tk.END,
                    values=(f"Cage {cage}", f"Rat {animal}", condition),
                    tags=(condition.upper() if condition != "Unassigned" else "UNASSIGNED",),
                )
                item_subjects[item_id] = (dataset, cage, animal)

        def apply_condition(condition: str | None) -> None:
            selected = tree.selection()
            if not selected:
                messagebox.showwarning("No subjects selected", "Select one or more Cage/Rat subjects first.", parent=window)
                return
            by_dataset: dict[str, list[tuple[str, str]]] = {}
            for item_id in selected:
                dataset, cage, animal = item_subjects[item_id]
                by_dataset.setdefault(dataset, []).append((cage, animal))
            try:
                for dataset, selected_subjects in by_dataset.items():
                    self.condition_map_store.update_many(dataset, selected_subjects, condition)
                    assignments[dataset] = self.condition_map_store.load(dataset)
            except (OSError, ValueError) as exc:
                messagebox.showerror("Could not save condition labels", str(exc), parent=window)
                return
            refresh_note = ""
            try:
                refreshed = [
                    self.results_workbook.refresh_tail_angle_consistency(dataset)
                    for dataset in by_dataset
                ]
                if any(path is not None for path in refreshed):
                    refresh_note = " Tail-angle graphs refreshed."
            except (OSError, ResultsWorkbookError) as exc:
                refresh_note = f" Group label saved; tail-angle chart refresh needs retry: {exc}"
            refresh_table()
            label = condition if condition else "Unassigned"
            self.status_var.set(f"Assigned {len(selected)} subject(s) as {label}.{refresh_note}")

        refresh_table()
        buttons = ttk.Frame(outer)
        buttons.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(buttons, text="Label selected SHAM", command=lambda: apply_condition("SHAM")).pack(
            side=tk.LEFT, expand=True, fill=tk.X
        )
        ttk.Button(buttons, text="Label selected STROKE", command=lambda: apply_condition("STROKE")).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=6
        )
        ttk.Button(buttons, text="Clear label", command=lambda: apply_condition(None)).pack(
            side=tk.LEFT, expand=True, fill=tk.X
        )
        ttk.Button(outer, text="Close", command=window.destroy).pack(anchor="e", pady=(8, 0))

    def _build_detection_controls(self, sidebar: ttk.Frame) -> None:
        box = ttk.LabelFrame(sidebar, text="DeepLabCut Tracking", padding=6)
        box.grid(row=7, column=0, sticky="ew", pady=(0, 8))
        box.columnconfigure(0, weight=1)

        self.detection_enabled_var.set(True)
        display_options = ttk.Frame(box)
        display_options.grid(row=0, column=0, sticky="ew")
        ttk.Checkbutton(
            display_options,
            text="Show live tracking",
            variable=self.detection_enabled_var,
            command=lambda: self.show_frame(self.current_frame),
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            display_options,
            text="Show tail angle",
            variable=self.angle_overlay_var,
            command=lambda: self.show_frame(self.current_frame),
        ).pack(side=tk.LEFT, padx=(8, 0))
        analysis_row = ttk.Frame(box)
        analysis_row.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        analysis_row.columnconfigure(0, weight=1)
        analysis_row.columnconfigure(1, weight=1)
        ttk.Button(
            analysis_row,
            text="Analyze Trial",
            command=self.analyze_current_tracking,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(
            analysis_row,
            text="Analyze Animal",
            command=self.analyze_selected_animal,
        ).grid(row=0, column=1, sticky="ew", padx=(3, 0))
        ttk.Button(
            box,
            text="Analyze Selected Day",
            command=self.analyze_selected_day,
        ).grid(row=2, column=0, sticky="ew", pady=(4, 0))



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
            row=2, column=0, sticky="ew", pady=(6, 0), padx=(0, 3)
        )
        ttk.Button(tick_box, text="Check Tickmarks", command=self.open_tickmark_checker).grid(
            row=2, column=1, sticky="ew", pady=(6, 0), padx=(3, 0)
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

    def open_tickmark_checker(self) -> None:
        """Review the first frame and applicable tick calibration for each trial."""
        if self.dataset is None or not self.day_var.get():
            self.status_var.set("Choose a dataset and day before checking tickmarks.")
            return

        day = self.day_var.get()
        videos = tick_review_videos(self.dataset, day)
        if not videos:
            self.status_var.set(f"No eligible trial videos found for {day}.")
            return

        initial_index = 0
        if self.current_video is not None:
            for index, video in enumerate(videos):
                if video.relative_path == self.current_video.relative_path:
                    initial_index = index
                    break

        window = tk.Toplevel(self)
        window.title(f"Check Tickmarks - {day}")
        window.transient(self)
        window.minsize(680, 520)

        outer = ttk.Frame(window, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)
        title_var = tk.StringVar()
        detail_var = tk.StringVar()
        ttk.Label(outer, textvariable=title_var, font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        ttk.Label(outer, textvariable=detail_var, wraplength=980).pack(anchor="w", pady=(2, 8))
        canvas = tk.Canvas(outer, bg="#000000", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(outer)
        controls.pack(fill=tk.X, pady=(8, 0))
        previous_button = ttk.Button(controls, text="← Previous")
        previous_button.pack(side=tk.LEFT)
        next_button = ttk.Button(controls, text="Next →")
        next_button.pack(side=tk.RIGHT)
        ttk.Label(controls, text="Use ← / → to move through trials and subjects.").pack(side=tk.LEFT, padx=12)

        state: dict[str, object] = {"index": initial_index, "photo": None}
        subject_order: list[str] = []
        for video in videos:
            if video.subject_key not in subject_order:
                subject_order.append(video.subject_key)

        def calibration_for_review(video: TrialVideo) -> tuple[BeamCalibration | None, bool]:
            # Show a currently edited draft in the review window as well, without
            # mistaking it for a confirmed calibration.
            if video == self.current_video and self.pending_calibration is not None:
                return self.pending_calibration, False
            confirmed = self.calibration_for_video(video)
            if confirmed is not None:
                return confirmed, True
            return self.tick_draft_for_video(video), False

        def read_first_frame(video: TrialVideo):
            capture = cv2.VideoCapture(str(video.path))
            try:
                ok, frame = capture.read()
            finally:
                capture.release()
            return frame if ok else None

        def render() -> None:
            index = int(state["index"])
            video = videos[index]
            calibration, confirmed = calibration_for_review(video)
            subject_index = subject_order.index(video.subject_key) + 1
            title_var.set(
                f"{day}  |  Cage {video.cage_number} Rat {video.rat_id}  |  "
                f"T{video.trial}  |  Subject {subject_index}/{len(subject_order)}"
            )

            frame = read_first_frame(video)
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    frame, "Could not read first frame", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
                )

            if calibration is None:
                detail_var.set("No tick calibration for this trial. It is not ready for analysis.")
                color = (80, 80, 255)
            else:
                label = "Confirmed" if confirmed else "Unconfirmed draft"
                source = f" from T{calibration.source_trial}" if confirmed else ""
                detail_var.set(
                    f"{label}{source}: {len(calibration.ticks)}/{len(BEAM_TICK_MARKS_CM)} ticks. "
                    + ("" if confirmed else "Unconfirmed ticks are not used for analysis.")
                )
                color = (199, 255, 92) if confirmed else (102, 224, 255)
                for tick in calibration.ticks:
                    x, y = int(tick.x), int(tick.y)
                    cv2.line(frame, (x, y - 14), (x, y + 14), color, 2, cv2.LINE_AA)
                    cv2.circle(frame, (x, y), 3, color, 2, cv2.LINE_AA)
                    cv2.putText(
                        frame, str(tick.distance_cm), (x + 5, y - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
                    )

            height, width = frame.shape[:2]
            scale = min(1.0, 980 / max(width, 1), 620 / max(height, 1))
            target_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            if target_size != (width, height):
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            state["photo"] = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            canvas.configure(width=target_size[0], height=target_size[1])
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=state["photo"])
            previous_button.configure(state="normal" if index else "disabled")
            next_button.configure(state="normal" if index < len(videos) - 1 else "disabled")

        def move(delta: int) -> str:
            next_index = max(0, min(int(state["index"]) + delta, len(videos) - 1))
            if next_index != state["index"]:
                state["index"] = next_index
                render()
            return "break"

        previous_button.configure(command=lambda: move(-1))
        next_button.configure(command=lambda: move(1))
        window.bind("<Left>", lambda _event: move(-1))
        window.bind("<Right>", lambda _event: move(1))
        window.bind("<Escape>", lambda _event: window.destroy())
        render()
        window.focus_set()

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

        ttk.Button(mark_box, text="Save Trials", command=self.save_annotation).grid(
            row=2, column=0, sticky="ew", pady=(7, 0), padx=(0, 3)
        )
        ttk.Button(mark_box, text="Save All Trials", command=self.save_all_trials).grid(
            row=2, column=1, sticky="ew", pady=(7, 0), padx=(3, 0)
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
        self.slider.bind("<Button-1>", self.on_slider_click, add="+")
        self.slider.bind("<ButtonRelease-1>", self.commit_seek, add="+")
        self.slider.bind("<KeyRelease>", self.commit_seek, add="+")
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
        self.guided_tick_sequence = False
        self.pending_seek_frame = None
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
        self.terminal_event = "manual"
        if row:
            outcome = normalize_outcome(row.get("outcome", ""))
            distance_cm = normalize_distance_cm(row.get("distance_cm", distance_cm))
            self.terminal_event = str(row.get("terminal_event", "manual")).strip() or "manual"

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

    def tick_draft_for_video(self, trial_video: TrialVideo) -> BeamCalibration | None:
        """Return a retained automatic draft without treating it as confirmed."""
        trial_key = calibration_key(trial_video, trial_specific=trial_video.trial != 1)
        return (
            self.saved_tick_drafts.get(trial_key)
            or self.saved_tick_drafts.get(calibration_key(trial_video))
        )

    def load_tick_calibration(self, trial_video: TrialVideo) -> None:
        self.active_tick_distance_cm = None
        self.guided_tick_sequence = False
        self.pending_calibration = None
        self.current_calibration = self.calibration_for_video(trial_video)
        if self.current_calibration is not None:
            self.tick_status_var.set(
                f"Confirmed T{self.current_calibration.source_trial} ticks loaded."
            )
            return

        draft = self.tick_draft_for_video(trial_video)
        if draft is not None:
            self.pending_calibration = draft
            self.tick_status_var.set(
                f"Unconfirmed automatic draft loaded: {len(draft.ticks)}/"
                f"{len(BEAM_TICK_MARKS_CM)} non-overlapping ticks. "
                "Place the missing ticks, then confirm intervals."
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
        missing_ticks = sorted(set(BEAM_TICK_MARKS_CM) - {tick.distance_cm for tick in calibration.ticks})
        if missing_ticks:
            missing_text = ", ".join(str(distance) for distance in missing_ticks)
            self.tick_status_var.set(
                f"Calibration remains unconfirmed. Place the missing tick(s): {missing_text} cm."
            )
            messagebox.showwarning(
                "Incomplete ticks",
                "All 0-120 cm tick marks must be placed before confirming.\n\n"
                f"Missing: {missing_text} cm",
            )
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
        self.tick_draft_store.delete(confirmed.key)
        self.saved_tick_drafts = self.tick_draft_store.load_by_key()
        self.current_calibration = confirmed
        self.pending_calibration = None
        self.active_tick_distance_cm = None
        self.guided_tick_sequence = False
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
        # A new calibration, or a missing point in a retained draft, can flow
        # through the remaining missing tick distances without re-selecting them.
        existing_distances = {tick.distance_cm for tick in calibration.ticks}
        self.guided_tick_sequence = not bool(calibration.ticks) or distance_cm not in existing_distances
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
        # Existing saved results own their start/stop marks. Otherwise rebuild
        # the automatic result from the stored tracking so reopening an already
        # analyzed trial does not require another DLC run.
        if not self.marks:
            self._apply_automatic_current_result()
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
        # Dragging the thumb can generate hundreds of intermediate values. Keep
        # only the target so the actual video seek happens once on release.
        self.pending_seek_frame = int(round(float(value)))

    def on_slider_click(self, event: tk.Event) -> None:
        """Jump immediately to the frame beneath a click on the slider track."""
        if self.cap is None or self.frame_count <= 1:
            return
        self.pause()
        first_x, _first_y = self.slider.coords(0)
        last_x, _last_y = self.slider.coords(self.frame_count - 1)
        span = last_x - first_x
        if span <= 0:
            return
        fraction = max(0.0, min(1.0, (event.x - first_x) / span))
        target = int(round(fraction * (self.frame_count - 1)))
        self.pending_seek_frame = None
        self.show_frame(target)

    def commit_seek(self, _event: tk.Event | None = None) -> None:
        """Render the final slider position after a drag or keyboard seek."""
        if self.cap is None:
            return
        target = self.pending_seek_frame
        self.pending_seek_frame = None
        if target is None:
            target = int(round(float(self.slider.get())))
        self.pause()
        self.show_frame(target)

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
        next_distance = (
            next_missing_guided_tick_distance(self.pending_calibration, distance_cm)
            if self.guided_tick_sequence
            else None
        )
        if next_distance is not None:
            self.active_tick_distance_cm = next_distance
            self.tick_edit_distance_var.set(next_distance)
            self.tick_status_var.set(
                f"Placed {distance_cm} cm tick at ({x},{y}). Now click the {next_distance} cm tick mark."
            )
        else:
            self.active_tick_distance_cm = None
            if self.guided_tick_sequence:
                self.guided_tick_sequence = False
                self.tick_status_var.set(
                    "Placed all remaining tick marks. Review the markers, then click Confirm Intervals to save."
                )
            else:
                self.tick_status_var.set(
                    f"Moved {distance_cm} cm tick to ({x},{y}). Confirm Intervals to save."
                )
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
            terminal_event=self.terminal_event,
        )

        try:
            self.annotation_store.save(annotation)
            self.results_workbook.save(annotation)
        except (OSError, ResultsWorkbookError) as exc:
            self.status_var.set(f"Saved annotation, but Excel could not be written: {exc}")
            return

        angle_trials, angle_frames, angle_skipped, angle_error, sd_graphs = self._export_tail_angle_batch(
            [self.current_video]
        )

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
        angle_message = (
            f" {angle_frames} valid tail-angle frame(s) exported."
            if angle_trials
            else ""
        )
        if sd_graphs:
            angle_message += " Tail-angle graphs refreshed."
        if angle_skipped:
            angle_message += " Tail-angle export skipped: tracking or confirmed ticks are unavailable."
        if angle_error:
            angle_message += f" Tail-angle export failed: {angle_error}"
        self.status_var.set(
            f"Saved trial result: {crossing:.2f} sec, distance {distance_cm} cm."
            f"{tail_message}{angle_message}"
        )

    def save_all_trials(self) -> None:
        """Save automatic results for T1-T3 of the selected mouse without DLC reruns."""
        if self.dataset is None:
            self.status_var.set("Choose a video folder first.")
            return
        subject_key = self.selected_subject_key()
        videos = (
            list(self.dataset.trials_for_subject(subject_key).values())
            if subject_key
            else []
        )
        if not videos:
            self.status_var.set("Select a mouse with available trials first.")
            return

        saved, skipped, graph_ok = self._save_auto_batch(videos, refresh_tail=True)
        angle_trials, angle_frames, angle_skipped, angle_error, sd_graphs = (
            self._export_tail_angle_batch(videos)
        )
        error_message = (
            f" {self.last_batch_save_error}"
            if skipped and self.last_batch_save_error
            else ""
        )
        graph_message = " Tail graph refreshed." if graph_ok else " Tail graph needs review."
        angle_message = (
            f" {angle_frames} valid tail-angle frame(s) exported from {angle_trials} trial(s)."
            if angle_trials
            else ""
        )
        if sd_graphs:
            angle_message += " Tail-angle graphs refreshed."
        if angle_skipped:
            angle_message += f" {angle_skipped} tail-angle export trial(s) skipped."
        if angle_error:
            angle_message += f" Tail-angle export failed: {angle_error}"
        self.status_var.set(
            f"Saved {saved} trial(s); {skipped} need review."
            f"{error_message}{graph_message}{angle_message}"
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
            / "RBT_front_back_body_tailS_trailE-RBT_CV-2026-08-17"
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

    def _save_tick_draft(self, video: TrialVideo, detection) -> bool:
        """Persist partial, well-spaced tick predictions separately from confirmations."""
        if not detection.ticks:
            return False
        draft = calibration_from_detection(
            video,
            detection,
            confirmed_at="",
            trial_specific=video.trial != 1,
        )
        self.tick_draft_store.save(draft)
        self.saved_tick_drafts = self.tick_draft_store.load_by_key()
        return True

    def auto_detect_ticks(self) -> None:
        if self.current_video is None:
            self.tick_status_var.set("Open a trial before detecting ticks.")
            return

        video = self.current_video
        def complete(succeeded: bool, message: str) -> None:
            if not succeeded:
                self.load_tick_calibration(video)
                draft_loaded = (
                    self.current_calibration is None
                    and self.pending_calibration is not None
                )
                if draft_loaded:
                    self.tick_status_var.set(
                        f"Tick scan incomplete. Kept {len(self.pending_calibration.ticks)}/"
                        f"{len(BEAM_TICK_MARKS_CM)} ticks (yellow). Add the missing ticks, then confirm."
                    )
                else:
                    self.tick_status_var.set(
                        "Tick scan incomplete. Use Set from click to place the ticks manually."
                    )
                if video == self.current_video:
                    self.refresh_frame()
                return
            self.load_tick_calibration(video)
            self.tick_status_var.set(
                "Tick Calibration Complete. 13/13 ticks found."
            )
            if video == self.current_video:
                self.refresh_frame()

        self._detect_ticks_with_fallback(video, complete, "Detecting Trial's Ticks")

    def _detect_ticks_with_fallback(
        self,
        video: TrialVideo,
        on_complete: Callable[[bool, str], None],
        label: str,
    ) -> None:
        """Try a fast distributed tick sample, then a broader one on failure."""
        _python, _runner, _tracking_config, tick_config = self._runtime()

        def attempt(frame_count: int, retrying: bool) -> None:
            scan_label = "broader" if retrying else "initial"
            self._run_dlc(
                [
                    "tick-analyze",
                    "--config",
                    str(tick_config),
                    "--video",
                    str(video.path),
                    "--early-frames",
                    str(frame_count),
                ],
                lambda code, output: finish(code, output, retrying),
                f"{label} ({scan_label} {frame_count}-frame scan)...",
            )

        def finish(code: int, output: str, retrying: bool) -> None:
            if code:
                on_complete(False, output.splitlines()[-1] if output else "Tick model failed.")
                return
            detection = self.dlc_tick_detector.detect_for_video(video)
            if self._save_calibration(video, detection):
                on_complete(True, detection.message)
                return
            if not retrying:
                self.tick_status_var.set(
                    f"No usable calibration from {INITIAL_TICK_SCAN_FRAMES} sampled frames; "
                    f"retrying with {BROADER_TICK_SCAN_FRAMES} frames..."
                )
                attempt(BROADER_TICK_SCAN_FRAMES, True)
                return
            self._save_tick_draft(video, detection)
            on_complete(False, detection.message)

        attempt(INITIAL_TICK_SCAN_FRAMES, False)

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
        elif timeline.ended_at_video_end:
            final_state = timeline.state_at(timeline.end_frame)
            if (
                final_state is not None
                and final_state.back_paw_x is not None
                and final_state.back_paw_y is not None
            ):
                stop_x = int(round(final_state.back_paw_x))
                stop_y = int(round(final_state.back_paw_y))

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
                terminal_event=(
                    "fall"
                    if outcome == OUTCOME_FELL
                    else "video_end"
                    if timeline.ended_at_video_end
                    else "tick_120"
                ),
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

        def run(index: int = 0, complete: int = 0, failed: list[str] | None = None) -> None:
            failed = failed or []
            if index >= len(t1_videos):
                self.saved_tick_calibrations = self.tick_store.load_by_key()
                failure_note = (
                    f" Manual calibration needed: {', '.join(failed)}."
                    if failed
                    else ""
                )
                self.tick_status_var.set(
                    f"Tick Calibration Complete. {complete}/{len(t1_videos)} "
                    f"T1 calibration(s) confirmed.{failure_note}"
                )
                if on_complete is not None:
                    on_complete(complete, len(t1_videos))
                return

            video = t1_videos[index]
            def finish(succeeded: bool, _message: str) -> None:
                next_failed = failed + (
                    [] if succeeded else [f"Cage {video.cage_number} Mouse {video.rat_id} T1"]
                )
                run(index + 1, complete + int(succeeded), next_failed)

            self._detect_ticks_with_fallback(
                video,
                finish,
                (
                    f"Detecting day ticks {index + 1}/{len(t1_videos)}: "
                    f"Cage {video.cage_number} Rat {video.rat_id}, T1"
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
            # A trial can have a perfectly valid 0 cm start but no detected fall
            # or 120 cm endpoint. Preserve that start marker for playback instead
            # of hiding it merely because there is no complete result to save.
            # It deliberately remains unsaveable until a terminal event exists.
            self._show_detected_start_for_incomplete_trial()
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
        self.terminal_event = annotation.terminal_event
        self.set_distance_cm(annotation.distance_cm)
        self.update_distance_controls()
        self.update_result_labels()

    def _show_detected_start_for_incomplete_trial(self) -> None:
        """Show a reliable automatic 0 cm start even without a terminal event."""
        if self.current_video is None or "stop" in self.marks:
            return

        if (
            self.tracking_video_path != self.current_video.path
            or self.current_scoring_tracking is None
        ):
            if not self._load_tracking(self.current_video):
                return

        calibration = self.calibration_for_video(self.current_video)
        if calibration is None or self.current_scoring_tracking is None:
            return

        try:
            timeline = analyze_tracking_timeline(
                self.current_scoring_tracking,
                calibration,
            )
        except ValueError:
            return
        if timeline.start_frame is None:
            return

        start_tick = point_for_distance(calibration, 0)
        if start_tick is None:
            return

        self.marks["start"] = PawMark(
            timeline.start_frame,
            start_tick.x,
            start_tick.y,
        )
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
                "0 cm start"
                if kind == "start"
                else "video end"
                if self.terminal_event == "video_end"
                else "120 cm end"
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
        tail_angle = None
        if (
            self.detection_enabled_var.get()
            and self.angle_overlay_var.get()
            and self.current_video is not None
        ):
            scoring_prediction = (
                self.current_scoring_tracking.points_for_frame(frame_number)
                if self.current_scoring_tracking is not None
                else None
            )
            tail_angle = calculate_tail_angle(
                scoring_prediction,
                self.calibration_for_video(self.current_video),
            )
        display = (
            draw_tracking_overlay(frame, prediction, tail_angle=tail_angle)
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

    def _export_tail_angle_batch(self, videos: list[TrialVideo]) -> tuple[int, int, int, str, int]:
        """Export frame-wise research angles without relying on legacy scoring."""
        exported_videos: list[TrialVideo] = []
        angle_records = []
        skipped = 0

        for video in videos:
            calibration = self.calibration_for_video(video)
            csv_path = self.tracking_store.find_for_video(video)
            if calibration is None or csv_path is None:
                skipped += 1
                continue
            try:
                tracking = self.tracking_store.load(csv_path).filtered(
                    SCORING_LIKELIHOOD_CUTOFF
                )
            except (OSError, ValueError):
                skipped += 1
                continue

            capture = cv2.VideoCapture(str(video.path))
            try:
                fps = float(capture.get(cv2.CAP_PROP_FPS) or 15.0)
            finally:
                capture.release()

            exported_videos.append(video)
            angle_records.extend(
                tail_angle_frame_records(video, tracking, calibration, fps)
            )

        error = ""
        if exported_videos:
            try:
                # The end-of-action refresh below updates the graph even if no
                # video could be exported, while avoiding a second redraw here.
                self.results_workbook.save_tail_angle_measurements(
                    exported_videos,
                    angle_records,
                    refresh_plot=False,
                )
            except (OSError, ResultsWorkbookError) as exc:
                error = str(exc)

        refreshed_graphs = 0
        for dataset in sorted({video.dataset for video in videos}):
            try:
                if self.results_workbook.refresh_tail_angle_consistency(dataset) is not None:
                    refreshed_graphs += 1
            except (OSError, ResultsWorkbookError) as exc:
                graph_error = str(exc)
                error = f"{error} {graph_error}".strip()
        return len(exported_videos), len(angle_records), skipped, error, refreshed_graphs

    def _analyze(
        self,
        videos: list[TrialVideo],
        label: str,
        refresh_tail: bool = False,
    ) -> None:
        if not videos:
            self.status_var.set("No videos are available for analysis.")
            return
        missing_calibrations = [
            video for video in videos if self.calibration_for_video(video) is None
        ]
        if missing_calibrations:
            requirements = tick_confirmation_requirements(
                missing_calibrations,
                self.saved_tick_calibrations,
            )
            details = "\n".join(f"- {requirement}" for requirement in requirements)
            self.status_var.set("Ticks need confirmation: " + "; ".join(requirements))
            messagebox.showwarning(
                "Tick confirmation required",
                f"{label.title()} cannot start until these tick intervals are confirmed:\n\n"
                f"{details}\n\n"
                "Confirm the listed T1 calibration(s); each one covers that mouse's T1-T3 trials.",
            )
            return

        _python, _runner, tracking_config, _tick_config = self._runtime()
        args = [
            "analyze-files",
            "--config",
            str(tracking_config),
            "--shuffle",
            "2",
        ]
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
        angle_trials, angle_frames, angle_skipped, angle_error, sd_graphs = self._export_tail_angle_batch(videos)
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
        angle_message = (
            f" {angle_frames} valid angle frame(s) exported from {angle_trials} trial(s) to the Frame Angles sheet."
            if angle_trials
            else ""
        )
        if sd_graphs:
            angle_message += " Tail-angle graphs refreshed."
        if angle_skipped:
            angle_message += f" {angle_skipped} angle export trial(s) skipped."
        if angle_error:
            angle_message += f" Angle export failed: {angle_error}"
        self.status_var.set(
            f"Analysis complete. {saved} result(s) saved to Excel; "
            f"{skipped} need review.{error_message}{graph_message}{angle_message}"
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


def _numeric_id_sort(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value.casefold())


def tick_confirmation_requirements(
    videos: list[TrialVideo],
    calibrations: Mapping[str, BeamCalibration],
) -> list[str]:
    """Return actionable shared-T1 calibration requirements for missing videos."""
    by_subject: dict[tuple[str, str, str, str], list[TrialVideo]] = {}
    for video in videos:
        by_subject.setdefault(
            (video.dataset, video.day, video.cage_number, video.rat_id), []
        ).append(video)

    requirements: list[str] = []
    for _dataset, _day, cage, animal in sorted(
        by_subject,
        key=lambda key: (_numeric_id_sort(key[2]), _numeric_id_sort(key[3])),
    ):
        subject_videos = by_subject[(_dataset, _day, cage, animal)]
        t1_video = next((video for video in subject_videos if video.trial == 1), subject_videos[0])
        shared_key = calibration_key(t1_video)
        if shared_key not in calibrations:
            requirements.append(f"Cage {cage} Mouse {animal}: T1 (covers T1-T3)")
            continue
        missing_trials = ", ".join(f"T{video.trial}" for video in sorted(subject_videos, key=lambda item: item.trial))
        requirements.append(f"Cage {cage} Mouse {animal}: {missing_trials}")
    return requirements


def tick_review_videos(dataset: DatasetIndex, day: str) -> list[TrialVideo]:
    """Order a day's eligible videos as T1-T3 for each subject."""
    videos: list[TrialVideo] = []
    for subject_key, _label in dataset.subjects_for_day(day):
        trials = dataset.trials_for_subject(subject_key)
        videos.extend(trials[trial] for trial in sorted(trials))
    return videos


def next_guided_tick_distance(distance_cm: int) -> int | None:
    """Return the next manual 10 cm tick, stopping after the 120 cm endpoint."""
    next_distance = distance_cm + 10
    return next_distance if next_distance <= BEAM_LENGTH_CM else None


def next_missing_guided_tick_distance(
    calibration: BeamCalibration,
    distance_cm: int,
) -> int | None:
    """Return the next unplaced 10 cm tick after a manual click."""
    placed = {tick.distance_cm for tick in calibration.ticks}
    next_distance = next_guided_tick_distance(distance_cm)
    while next_distance is not None:
        if next_distance not in placed:
            return next_distance
        next_distance = next_guided_tick_distance(next_distance)
    return None


def check_app() -> int:
    dataset = DatasetIndex()
    print(f"Dataset: {dataset.dataset_dir.relative_to(ROOT)}")
    print(f"All videos: {len(dataset.all_videos)}")
    print(f"Review videos: {len(dataset.videos)}")
    eligible = ", ".join(
        f"Cage {cage} Rat {rat_id}"
        for _dataset, cage, rat_id in sorted(dataset.evaluated_subject_keys)
    )
    print(f"BL/D30 eligible roster: {eligible or 'not found; showing no subjects'}")
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
