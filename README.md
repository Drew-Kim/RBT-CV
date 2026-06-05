# RBT-CV

Python tooling for reviewing rotating beam test videos, marking hind-paw timing events, and comparing computed crossing times against the corrected manual workbook.

## Setup

Create the local environment from PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m rbtcv.app --check
```

The app auto-detects the available dataset folder under `data/`, preferring `RBT DATA` and then `RBT DATA_B`.

For the current test set, the GUI uses the `D30` folder as the survivor roster. A video named like `D10_2_T1_...avi` means cage `10`, rat ID `2`, trial `1`. Only mice that appear in `D30` are included for review, and their matching videos from earlier days are included by cage number plus rat ID even when the filename prefix changes, such as `C10_2` on earlier days and `D10_2` on day 30.

## Run the GUI

```powershell
.\.venv\Scripts\python -m rbtcv.app
```

The GUI writes annotation results to `outputs/annotations.csv` when timing marks are saved. If `outputs/manual_mapping.csv` exists, it will be used for workbook matching.
