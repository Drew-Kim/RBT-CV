# RBT-CV

RBT-CV is the review station for rotating-beam trials. It keeps the video, beam calibration, DeepLabCut tracking, automatic scoring, and saved result in one place. The model records a completed automatic result after analysis; you can still review and overwrite it when needed.

The tracking model uses four landmarks:

- `visible_back_paw`
- `visible_front_paw`
- `tail_end`
- `body_center`

Only the back paw and body center drive the automatic score. The front paw and tail end are shown so you can judge whether the tracking is believable.

## What the app expects

Keep the video folders below `data`:

```text
data\
  RBT DATA_B\
  SEONG RBT DATA\
```

`RBT DATA_B` is the regular review set. `SEONG RBT DATA` stays available because it contains useful fall examples. The application opens maximized, with the video on the right and a scrollable control panel on the left.

The normal Python environment runs the review window. DeepLabCut stays in its own `.venv-dlc` environment.

## Start the review window

From PowerShell in the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m rbtcv.app
```

If DeepLabCut has not been set up on this computer, run this once before trying any of the Analyze buttons:

```powershell
.\scripts\setup_dlc_venv.ps1
```

## The normal review routine

This is the order to use for a new animal/day. It avoids mixing a trial's tracking with an unconfirmed calibration.

### 1. Choose the dataset, then open T1

1. The app starts blank. Click **Choose** and select either the `data` folder or one dataset folder inside it.
2. Use the **Day** drop-down to select `BL`, `D3`, `D8`, and so on.
3. Select the cage/rat from **Subjects**. T1 opens automatically.
4. Use **T1**, **T2**, or **T3** to change trials. Disabled buttons mean that trial is not available for the selected animal.

The **Trials per subject** control only changes how many trial buttons are enabled. It does not remove files from the dataset.

### 2. Calibrate the beam on T1

Calibration is done on T1 and shared with T2 and T3 for the same cage, rat, and day. Do not calibrate each trial separately.

There are two detection choices:

- **Detect Trial's Ticks** runs the tick model only on the open animal's T1. It samples 10 frames across the full trial, keeps up to the five clearest non-overlapping predictions, and uses their median for each tick. If only one or two frames are clear, they are still used rather than blending in a bad frame. A calibration with merged or out-of-order ticks is rejected instead of being reshaped. Review the proposed marks, correct an individual mark if needed, then use **Confirm Intervals**.
- **Detect Day's Ticks** runs the model on every animal's T1 for the selected day. Each complete T1 calibration is saved automatically and shared with that animal's T2 and T3. It never analyzes T2 or T3 for tick marks.

Use **Show tick marks** to see the marks. If one is wrong, choose its distance in **Edit** and click **Set from click**, then click the correct mark in the video. When the marks are right, click **Confirm Intervals**.

The 0 cm tick is the timing start line. The 120 cm tick is the platform line. The orange line through the tick centers is also the fall boundary, so its placement matters.

### 3. Get tracking for the video

The **DeepLabCut Tracking** panel has three choices:

| Button | Use it when |
| --- | --- |
| **Analyze Current Trial** | You only need the video currently open. |
| **Analyze Selected Animal (T1-T3)** | You want all available trials for the selected cage/rat. |
| **Analyze Selected Day** | You are ready to process every review video for the selected day. |

Only one DeepLabCut job can run at a time. Progress and any error appear in the status line at the bottom of the window. New prediction CSVs are saved in `outputs\dlc_predictions` and load automatically. When analysis of the open trial finds a completed reach or fall, its result is saved automatically; incomplete runs stay available for review. The app also recognizes matching CSVs beside a source video or inside a DLC project's `videos` folder.

Check **Show live tracking (>= 0.20)** to see the colored landmarks during playback. The display threshold is intentionally forgiving so you can see what the model is doing. Scoring uses the same CSV at the stricter 0.60 threshold.

### 4. Watch the trial and check the result

Playback controls sit below the video:

- `|<` moves back about one second.
- `<` and `>` move one frame.
- **Play** / **Pause** starts and stops playback.
- The slider jumps to any frame.
- The frame number and elapsed video time are shown beside the slider.

Keyboard shortcuts are useful during review: **Space** plays/pauses, and the **Left** and **Right** arrow keys step one frame.

Automatic scoring follows the beam in this order:

1. The visible back paw must cross the vertical 0 cm tick in the direction of the platform. This starts the timer.
2. While the animal is running, the app keeps its farthest back-paw distance.
3. If the body center drops below the orange calibrated line, the trial is marked **Fell**. The farthest distance is retained and the scored time is set to **60.00 seconds**.
4. If the back paw crosses the vertical 120 cm tick first, the trial is marked **Reached platform** and the distance is set to 120 cm.

The overlay is a review aid, not proof by itself. Pay special attention to frames where the back paw is hidden, the animal turns around, or a fall occurs near the end of the beam.

### 5. Correct or save the outcome

The **Trial Outcome** panel is the final check before writing data:

- **Reached platform** always uses 120 cm; the distance field is locked.
- **Fell** unlocks the distance field, which uses 5 cm steps from 0 to 120.
- **Scoring Preview** shows the value that will be saved.

A completed automatic result is saved after analysis. You can correct an unusual result and click **Save Trial Result** again after checking the video, overlay, tick marks, and outcome. Saving the same video again updates its row rather than creating a duplicate.

Each save also updates `outputs\\RBT_CV_Results.xlsx`. Its **Forelimb** sheet follows the time-table layout in `res\\RBT_Data_Corrected.xlsb`, and the **RBT-CV Results** sheet keeps the full audit record (video, frames, outcome, distance, and saved time). The original `.xlsb` file is never changed. Close the output workbook in Excel before saving another trial.

Saved results are written to `outputs\annotations.csv`.

## Useful fall checks

For a quick fall test, choose day `D3`, then select the matching `SEONG RBT DATA` row in **Subjects** for one of these animals:

- Cage 1, Rat 5
- Cage 3, Rat 2
- Cage 3, Rat 3
- Cage 4, Rat 1

`C1_5_T2_2025-08-22-142736-0000.avi` already has a tracking CSV and is a convenient first check for the live overlay.

## If something does not look right

**No tracking dots:** Make sure the live-tracking box is checked. If no CSV is available, run **Analyze Current Trial** and wait for the status line to report completion. The dots will load automatically.

**The result does not appear automatic:** Confirm the 0-120 cm calibration first. Automatic timing needs both a valid calibration and a DLC CSV with confident back-paw/body-center points.

**The tick buttons refuse to work:** The single-animal detection, editing, and confirmation controls use T1 because its calibration is shared across all three trials. For the whole selected day, use **Detect Day's Ticks**; it analyzes only each animal's T1.

**The animal list is long:** Use the scrollbar inside **Subjects**. The left panel itself also scrolls if the screen is short.

## DeepLabCut model workflow

The command helper is `scripts\dlc_tracking.py`. A typical model workflow is:

```powershell
# Create a project from a small starting set.
.\.venv-dlc\Scripts\python scripts\dlc_tracking.py create

# Train after frames are labeled.
.\.venv-dlc\Scripts\python scripts\dlc_tracking.py train --config "models\dlc_tracking\YOUR_PROJECT\config.yaml"

# Try predictions on a small batch before processing a full day.
.\.venv-dlc\Scripts\python scripts\dlc_tracking.py analyze --config "models\dlc_tracking\YOUR_PROJECT\config.yaml" --day D30 --trial 1 --limit 3
```

When labeling, leave a landmark blank if it cannot be seen. For this task, a smaller set of varied, well-labeled frames is better than many near-identical frames at the platform.

## Files written by the application

| File or folder | Contents |
| --- | --- |
| `outputs\annotations.csv` | Final saved outcome, distance, frame times, and scored time for each trial. |
| `outputs\tick_calibrations.csv` | Confirmed beam ticks, shared by an animal/day. |
| `outputs\dlc_predictions\` | Mouse-tracking prediction CSVs created from the GUI. |
| `outputs\dlc_tick_predictions\` | Tick-model prediction CSVs used during calibration. |

## Quick checks

These do not open the GUI or require DeepLabCut:

```powershell
.\.venv\Scripts\python -m rbtcv.app --check
.\.venv\Scripts\python scripts\dlc_tracking.py test
```

The first command reports the indexed datasets and video count. The second checks the DLC CSV reader plus the automatic 0 cm start, fall penalty, farthest-distance, and 120 cm finish rules.
