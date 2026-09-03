# RBT-CV

RBT-CV is an ETL and analysis workflow for rotating beam trials under Dr.Vahdat's StrokeNeuroRecovery Lab. This program makes video analysis easier using DeepLabCut models for beam calibrations, automatic timing, fall detection inputing the results into an Excel results, and plotting a tail position graph.

The GUI uses 2 models, one for tracking four points on each mouse, the front paw, back paw, tail end, and body center. The back paw controls the start and finish of a trial. The body center is used for fall detection and fall-distance scoring. The tail end is used for creating the tail position graph for patterns. The second model is used for tickmark detection across each trial used during fall instances to record the maximum distance traveled by the mouse.

## Video Demo (Google Drive)

[GUI_Demo](https://drive.google.com/file/d/1IpKx7Tn9UOQS1KUdhf7VTKW0y6C3qlFg/view?usp=drive_link)

## Features

- DeepLabCut models for the front paw, back paw, tail end, tail middle, tail start, and body center and tickmark detection
- Automatic 0 cm start and 120 cm finish timing
- Fall detection with a 15px margin to avoid counting a brief hanging as a fall in order to account for recovery
- Tick calibration and the analysis can be run per trial, per animal, or per day
- The results recorded in a Excel tables for time, distance, and speed
- Normalized tail position graph per day, where each line represents the average tail positon for an animal across 3 trials

## Requirements

RBT-CV is set up for a Windows workstation. There are two separate Python environments because the review window and DeepLabCut do different jobs.

**For the GUI window**

- Windows 10 or 11
- Python 3.10
- A local `.venv` environment with the packages in `requirements.txt`: NumPy, OpenCV, Pillow, and OpenPyXL

**For DeepLabCut analysis**

- A local `.venv-dlc` environment created with `scripts\setup_dlc_venv.ps1`
- The trained mouse-tracking and tickmark DLC projects in the `models` folder
- A strong GPU is recommended for day analysis due to the large amount of video frames being evaluated.

The RBT videos, trained models, labeled frames, and generated results are excluded out of repo for lab confidentiality.

## Quick start

Open PowerShell in the RBT-CV folder.

**Environment setup**

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\scripts\setup_dlc_venv.ps1
```

**Starting the GUI**

```powershell
.\.venv\Scripts\python -m rbtcv.app
```

The app starts with a black video screen. Click **Choose**, select the RBT dataset folder, then choose the day, animal, and trial.

## Dataset and model locations

The GUI expects a dataset folder with day folders containing the trial videos. 
The trained models are expected to be inside the `models` folder

## Using the GUI

### Normal workflow

1. Click **Choose**, then select the dataset folder.
2. Choose a **Day**, then a subject and T1, T2, or T3.
3. Calibrate the beam ticks before tracking analysis. This is required for max distance during falling instances.
4. Analyze the open trial, selected animal, or selected day.
5. Check the scoring preview and Excel row if a result needs review.

For a full day, **Calibrate & Analyze Day** is the usual button to use. It calibrates T1 for each animal first, then analyzes the available T1-T3 videos and saves the results.

### Tick calibration

| Button | Functionality |
| --- | --- |
| **Detect Trial's Ticks** | Samples 10 frames across the open video, keeps clear nonoverlapping tickmarks, and uses their median confidence level for the position. |
| **Detect Day's Ticks** | Runs T1 tick calibration for every animal on the selected day. T1 is the shared calibration for that animal/day. |
| **Set from click** | Moves one selected tick to a manually clicked position. Use **Confirm Intervals** afterward. |
| **Calibrate & Analyze Day** | Runs the day calibration, confirms it, then starts day analysis. |

T2 and T3 use the T1 calibration by default. If one of those trials needs its own correction, detecting or manually editing ticks while it is open creates a trial-specific recalibration without changing T1.

### Tracking and scoring

Enable **Show live tracking (>= 0.20)** to inspect the model's tracking on the overlay while the video plays.

| Event | Rule | Results |
| --- | --- | --- |
| Start | First reliable backpaw point at or beyond the 0 cm tick | Green 0 cm marker and start frame |
| Finish | Back paw crosses 120 cm, or remains within 3 pixels of it for two frames | Green 120 cm marker and crossing time |
| Fall | Body center is more than 15 pixels below the calibrated beam boundary | Red mark on the body center, 60 second penalty, and farthest distance |

If a fall happens between two distance marks, the saved distance is the interval midpoint. A fall between 30 and 40 cm is recorded as 35 cm.

| Button | Functionality |
| --- | --- |
| **Analyze Current Trial** | Analyzes the video currently open in the viewer |
| **Analyze Selected Animal (T1-T3)** | Analyzes all available trials for the selected animal |
| **Analyze Selected Day** | Analyzes all review videos for the selected day |

## Results

Each dataset receives its own results folder. For example:

```text
outputs\SEONG RBT DATA Results\RBT_CV_Results.xlsx
outputs\SEONG RBT DATA Results\tail_position_measurements.csv
outputs\SEONG RBT DATA Results\tail_position_D3.svg
```

The "Forelimb" sheet has separate Time, Distance, and Speed tables. Rows stay in cage/animal order, and saving the same day, animal, and trial again overwrites the existing entry.

The tail graph has one normalized line per animal. A label such as "C9_2 (2/3 trials)" means two of that animal's three trials currently have usable tail position data. Positive values are above the fall boundary while negative values are below it.
