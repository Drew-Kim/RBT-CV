from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .annotations import AnnotationStore, TrialAnnotation, now_stamp
from .dataset import DatasetIndex, ROOT, TrialVideo
from .workbook import ManualMatch, ManualTimingStore


TIMING_TOLERANCE_SECONDS = 0.10


@dataclass
class PawMark:
    frame: int
    x: int
    y: int


class RBTReviewApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("RBT-CV Review")
        self.minsize(1020, 660)

        self.dataset = DatasetIndex()
        self.manual_store = ManualTimingStore()
        self.annotation_store = AnnotationStore()
        self.saved_annotations = self.annotation_store.load_by_video()

        self.cap: cv2.VideoCapture | None = None
        self.current_video: TrialVideo | None = None
        self.current_manual: ManualMatch | None = None
        self.current_frame = 0
        self.frame_count = 0
        self.fps = 15.0
        self.video_width = 640
        self.video_height = 480
        self.photo: ImageTk.PhotoImage | None = None
        self.playing = False
        self.after_id: str | None = None
        self.active_mark: str | None = None
        self.marks: dict[str, PawMark] = {}
        self.updating_slider = False

        self.day_var = tk.StringVar()
        self.trial_count_var = tk.IntVar(value=3)
        self.status_var = tk.StringVar(value="Ready")
        self.frame_var = tk.StringVar(value="Frame -")
        self.time_var = tk.StringVar(value="Time -")
        self.start_var = tk.StringVar(value="Start not marked")
        self.stop_var = tk.StringVar(value="Stop not marked")
        self.result_var = tk.StringVar(value="Crossing time -")
        self.manual_var = tk.StringVar(value="Manual time -")

        self.subject_labels: list[str] = []
        self.trial_buttons: dict[int, ttk.Button] = {}

        self._build_ui()
        self._bind_keys()
        self._populate_days()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=(10, 8))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Dataset").grid(row=0, column=0, sticky="w")
        self.dataset_label = ttk.Label(top, text=str(self.dataset.dataset_dir.relative_to(ROOT)))
        self.dataset_label.grid(row=0, column=1, sticky="w", padx=(8, 12))
        ttk.Button(top, text="Choose", command=self.choose_dataset).grid(row=0, column=2, padx=(0, 6))
        ttk.Button(top, text="Reload", command=self.reload_dataset).grid(row=0, column=3)

        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))

        sidebar = ttk.Frame(main, padding=(0, 0, 10, 0), width=260)
        main.add(sidebar, weight=0)

        video_panel = ttk.Frame(main)
        video_panel.columnconfigure(0, weight=1)
        video_panel.rowconfigure(0, weight=1)
        main.add(video_panel, weight=1)

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

        ttk.Label(sidebar, text="Subjects").grid(row=3, column=0, sticky="w")
        self.subject_list = tk.Listbox(sidebar, height=14, exportselection=False)
        self.subject_list.grid(row=4, column=0, sticky="nsew", pady=(2, 8))
        self.subject_list.bind("<<ListboxSelect>>", lambda _event: self.update_trial_buttons())
        sidebar.rowconfigure(4, weight=1)
        sidebar.columnconfigure(0, weight=1)

        trial_row = ttk.Frame(sidebar)
        trial_row.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        for trial in range(1, 4):
            button = ttk.Button(trial_row, text=f"T{trial}", command=lambda trial=trial: self.open_trial(trial))
            button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0 if trial == 1 else 4, 0))
            self.trial_buttons[trial] = button

        mark_box = ttk.LabelFrame(sidebar, text="Timing marks", padding=8)
        mark_box.grid(row=6, column=0, sticky="ew", pady=(0, 10))
        ttk.Button(mark_box, text="Mark start", command=lambda: self.begin_mark("start")).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(mark_box, text="Mark stop", command=lambda: self.begin_mark("stop")).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Label(mark_box, textvariable=self.start_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(mark_box, textvariable=self.stop_var).grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Button(mark_box, text="Save timing", command=self.save_annotation).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )
        mark_box.columnconfigure(0, weight=1)
        mark_box.columnconfigure(1, weight=1)

        result_box = ttk.LabelFrame(sidebar, text="Validation", padding=8)
        result_box.grid(row=7, column=0, sticky="ew")
        ttk.Label(result_box, textvariable=self.result_var).grid(row=0, column=0, sticky="w")
        ttk.Label(result_box, textvariable=self.manual_var).grid(row=1, column=0, sticky="w")

        self.canvas = tk.Canvas(video_panel, width=640, height=480, bg="#101010", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="n")
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        controls = ttk.Frame(video_panel, padding=(0, 8, 0, 0))
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

        status = ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(10, 4))
        status.grid(row=2, column=0, sticky="ew")

    def _bind_keys(self) -> None:
        self.bind("<space>", lambda _event: self.toggle_play())
        self.bind("<Left>", lambda _event: self.step_frame(-1))
        self.bind("<Right>", lambda _event: self.step_frame(1))

    def _populate_days(self) -> None:
        days = self.dataset.days
        self.day_combo["values"] = days
        if days:
            self.day_var.set(days[0])
            self.populate_subjects()
        else:
            self.status_var.set("No videos found.")

    def choose_dataset(self) -> None:
        selected = filedialog.askdirectory(initialdir=str(ROOT / "data"), title="Choose RBT dataset folder")
        if selected:
            self.dataset = DatasetIndex(Path(selected))
            self.dataset_label.config(text=str(self.dataset.dataset_dir.relative_to(ROOT)))
            self._populate_days()

    def reload_dataset(self) -> None:
        self.dataset = DatasetIndex(self.dataset.dataset_dir)
        self.saved_annotations = self.annotation_store.load_by_video()
        self.dataset_label.config(text=str(self.dataset.dataset_dir.relative_to(ROOT)))
        self._populate_days()
        self.status_var.set("Dataset reloaded.")

    def populate_subjects(self) -> None:
        self.subject_list.delete(0, tk.END)
        self.subject_labels = self.dataset.subjects_for_day(self.day_var.get())
        for label in self.subject_labels:
            self.subject_list.insert(tk.END, label)
        if self.subject_labels:
            self.subject_list.selection_set(0)
            self.subject_list.activate(0)
        self.update_trial_buttons()

    def selected_subject_key(self) -> str | None:
        selection = self.subject_list.curselection()
        if not selection:
            return None
        label = self.subject_labels[selection[0]]
        return self.dataset.subject_key_from_label(label)

    def update_trial_buttons(self) -> None:
        subject_key = self.selected_subject_key()
        available = self.dataset.trials_for_subject(subject_key) if subject_key else {}
        try:
            trial_count = int(self.trial_count_var.get())
        except tk.TclError:
            trial_count = 3

        for trial, button in self.trial_buttons.items():
            state = "normal" if trial <= trial_count and trial in available else "disabled"
            button.configure(state=state)

    def open_trial(self, trial: int) -> None:
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
        self.current_manual = self.manual_store.lookup(trial_video)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        self.canvas.config(width=self.video_width, height=self.video_height)
        self.slider.configure(to=max(self.frame_count - 1, 0))
        self.marks = self._loaded_marks_for(trial_video)
        self.current_frame = 0
        self.show_frame(0)
        self.update_mark_labels()
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

    def show_frame(self, frame_number: int) -> None:
        if self.cap is None:
            return
        frame_number = max(0, min(frame_number, max(self.frame_count - 1, 0)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = self.cap.read()
        if not ok:
            self.pause()
            return

        self.current_frame = frame_number
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(image=image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.draw_mark_overlays()

        self.updating_slider = True
        self.slider.set(frame_number)
        self.updating_slider = False
        self.frame_var.set(f"Frame {frame_number + 1}/{max(self.frame_count, 1)}")
        self.time_var.set(f"{self.frame_time(frame_number):.2f} sec")

    def draw_mark_overlays(self) -> None:
        colors = {"start": "#00d4ff", "stop": "#ff4d8d"}
        for kind, mark in self.marks.items():
            if mark.frame != self.current_frame:
                continue
            color = colors[kind]
            r = 6
            self.canvas.create_oval(mark.x - r, mark.y - r, mark.x + r, mark.y + r, outline=color, width=3)
            self.canvas.create_line(mark.x - 10, mark.y, mark.x + 10, mark.y, fill=color, width=2)
            self.canvas.create_line(mark.x, mark.y - 10, mark.x, mark.y + 10, fill=color, width=2)
            self.canvas.create_text(mark.x + 12, mark.y - 12, text=kind, fill=color, anchor="w")

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

    def begin_mark(self, kind: str) -> None:
        if self.cap is None:
            self.status_var.set("Load a trial before marking.")
            return
        self.pause()
        self.active_mark = kind
        self.status_var.set(f"Click the hind paw for the {kind} event on the current frame.")

    def on_canvas_click(self, event: tk.Event) -> None:
        if self.cap is None:
            return
        if self.active_mark is None:
            self.status_var.set("Choose Mark start or Mark stop, then click the hind paw.")
            return

        x = int(max(0, min(event.x, self.video_width - 1)))
        y = int(max(0, min(event.y, self.video_height - 1)))
        self.marks[self.active_mark] = PawMark(self.current_frame, x, y)
        self.status_var.set(f"Marked {self.active_mark} at frame {self.current_frame + 1}, x={x}, y={y}.")
        self.active_mark = None
        self.update_mark_labels()
        self.update_result_labels()
        self.show_frame(self.current_frame)

    def update_mark_labels(self) -> None:
        self.start_var.set(self._mark_text("start"))
        self.stop_var.set(self._mark_text("stop"))

    def _mark_text(self, kind: str) -> str:
        mark = self.marks.get(kind)
        if mark is None:
            return f"{kind.capitalize()} not marked"
        return f"{kind.capitalize()} F{mark.frame + 1} {self.frame_time(mark.frame):.2f}s ({mark.x},{mark.y})"

    def crossing_time(self) -> float | None:
        start = self.marks.get("start")
        stop = self.marks.get("stop")
        if start is None or stop is None:
            return None
        return (stop.frame - start.frame) / self.fps if self.fps else None

    def update_result_labels(self) -> None:
        crossing = self.crossing_time()
        if crossing is None:
            self.result_var.set("Crossing time -")
        elif crossing < 0:
            self.result_var.set("Crossing time invalid")
        else:
            self.result_var.set(f"Crossing time {crossing:.2f} sec")

        if self.current_manual is None:
            self.manual_var.set("Manual time not mapped")
            return

        if crossing is None or crossing < 0:
            self.manual_var.set(f"Manual time {self.current_manual.time_seconds:.2f} sec")
            return

        delta = crossing - self.current_manual.time_seconds
        status = "PASS" if abs(delta) <= TIMING_TOLERANCE_SECONDS else "CHECK"
        self.manual_var.set(f"Manual {self.current_manual.time_seconds:.2f} sec | delta {delta:+.2f} | {status}")

    def save_annotation(self) -> None:
        if self.current_video is None:
            self.status_var.set("Load a trial before saving.")
            return
        start = self.marks.get("start")
        stop = self.marks.get("stop")
        if start is None or stop is None:
            messagebox.showwarning("Missing marks", "Mark both start and stop before saving.")
            return
        crossing = self.crossing_time()
        if crossing is None or crossing < 0:
            messagebox.showwarning("Invalid timing", "Stop frame must be after start frame.")
            return

        manual_time = ""
        delta = ""
        validation = "not mapped"
        if self.current_manual is not None:
            manual_time = f"{self.current_manual.time_seconds:.4f}"
            delta_value = crossing - self.current_manual.time_seconds
            delta = f"{delta_value:.4f}"
            validation = "pass" if abs(delta_value) <= TIMING_TOLERANCE_SECONDS else "check"

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
            manual_time=manual_time,
            delta=delta,
            validation=validation,
            saved_at=now_stamp(),
        )
        self.annotation_store.save(annotation)
        self.saved_annotations = self.annotation_store.load_by_video()
        self.status_var.set(f"Saved timing: {crossing:.2f} sec ({validation}).")

    def destroy(self) -> None:
        self.pause()
        if self.cap is not None:
            self.cap.release()
        super().destroy()


def check_app() -> int:
    dataset = DatasetIndex()
    manual = ManualTimingStore()
    print(f"Dataset: {dataset.dataset_dir.relative_to(ROOT)}")
    print(f"All videos: {len(dataset.all_videos)}")
    print(f"Review videos: {len(dataset.videos)}")
    survivors = ", ".join(f"Cage {cage} Rat {rat_id}" for cage, rat_id in sorted(dataset.survivor_keys))
    print(f"D30 survivor roster: {survivors or 'not found; showing all videos'}")
    print(f"Days: {', '.join(dataset.days)}")
    print(f"Manual timing rows: {len(manual.times)}")
    print(f"Mapping file: {manual.mapping_path.relative_to(ROOT)} ({'found' if manual.mapping_path.exists() else 'not found'})")
    if manual.load_errors:
        print("Manual timing warnings:")
        for error in manual.load_errors:
            print(f"  {error}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RBT-CV video review GUI")
    parser.add_argument("--check", action="store_true", help="load data/workbook without opening the GUI")
    args = parser.parse_args(argv)

    if args.check:
        return check_app()

    app = RBTReviewApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
