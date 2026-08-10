from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rbtcv.detection import TRACKING_BODYPARTS
from extract_platform_end_frames import extract_phase

SOURCE_CONFIG = ROOT / "models" / "dlc_tracking" / "RBT_visible_front_back_tail-RBT_CV-2026-07-14" / "config.yaml"
PROJECT_DIR = ROOT / "models" / "dlc_tracking" / "RBT_visible_front_back_tail_endpoints_20x5-RBT_CV-2026-07-20"
MANIFEST = ROOT / "outputs" / "endpoints_20x5_mouse_tracking_batch.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a distinct 20-video endpoint supplement")
    parser.add_argument("--source-config", type=Path, default=SOURCE_CONFIG)
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    args = parser.parse_args()
    if not args.source_config.exists() or args.project_dir.exists():
        print("Source config is missing or destination already exists.")
        return 1
    import deeplabcut
    from deeplabcut.utils import auxiliaryfunctions
    source=auxiliaryfunctions.read_config(str(args.source_config))
    all_videos=[Path(path) for path in source.get("video_sets", {})]
    if len(all_videos)!=66 or any(not v.exists() for v in all_videos):
        print("Expected 66 active source videos.")
        return 1
    videos=[all_videos[round(i*(len(all_videos)-1)/19)] for i in range(20)]
    args.project_dir.mkdir(parents=True); (args.project_dir/'videos').mkdir(); config_path=args.project_dir/'config.yaml'
    try:
        shutil.copy2(args.source_config,config_path)
        config=auxiliaryfunctions.read_config(str(config_path)); config['Task']='RBT_visible_front_back_tail_endpoints_20x5'; config['project_path']=str(args.project_dir.resolve()); config['video_sets']={}; config['bodyparts']=list(TRACKING_BODYPARTS); config['TrainingFraction']=[0.95]; auxiliaryfunctions.write_config(str(config_path),config)
        deeplabcut.add_new_videos(str(config_path),[str(v) for v in videos],copy_videos=True)
        MANIFEST.parent.mkdir(parents=True,exist_ok=True)
        with MANIFEST.open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=('source_video','frames','sampling')); writer.writeheader(); writer.writerows({'source_video':str(v),'frames':5,'sampling':'80-92%: 2; 92-100%: 3'} for v in videos)
        extract_phase(deeplabcut,auxiliaryfunctions,config_path,videos,2,0.80,0.92)
        extract_phase(deeplabcut,auxiliaryfunctions,config_path,videos,3,0.92,1.00)
    except Exception:
        shutil.rmtree(args.project_dir,ignore_errors=True); raise
    finally:
        if config_path.exists():
            config=auxiliaryfunctions.read_config(str(config_path)); config['bodyparts']=list(TRACKING_BODYPARTS); config['TrainingFraction']=[0.95]; config['numframes2pick']=5; config['start']=0.0; config['stop']=1.0; auxiliaryfunctions.write_config(str(config_path),config)
    print(f'Created {args.project_dir.name}: 20 videos, target 100 endpoint frames.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())