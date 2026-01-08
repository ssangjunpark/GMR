import argparse
import pathlib
import os
import time
import csv

import numpy as np


from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

from rich import print

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_files_path",
        help="SMPLX motion filess to load.",
        type=str,
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openloong", "tienkung"],
        default="unitree_g1",
    )
    
    parser.add_argument(
        "--loop_count",
        default=1,
        type=int,
        help="Loop the motion.",
    )

    args = parser.parse_args()


    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"

    path = pathlib.Path(args.smplx_files_path)
    files = list(path.glob("*.npz"))
    # breakpoint() 

    log = "./results.csv"

    with open(log, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file_path', 'total_frames', 'total_time', 'avg_fps'])
    
    for file_path in files:
        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
            str(file_path), SMPLX_FOLDER
        )
        
        # align fps
        tgt_fps = 30
        smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
        
    
        # Initialize the retargeting system
        retarget = GMR(
            actual_human_height=actual_human_height,
            src_human="smplx",
            tgt_robot=args.robot,
        )

        
        total_frames = 0
        start_time = time.time()
        
        for loop_idx in range(args.loop_count):
            for i in range(len(smplx_data_frames)):
                smplx_data = smplx_data_frames[i]

                qpos = retarget.retarget(smplx_data)
                
                total_frames += 1
                
        end_time = time.time()
        total_time = end_time - start_time
        avg_fps = total_frames / total_time if total_time > 0 else 0
        print(f"file processed {file_path}")
        print(f"Total frames processed: {total_frames}")
        print(f"Total time taken: {total_time:.4f} seconds")
        print(f"Raw Retargeting FPS: {avg_fps:.2f}")

        with open(log, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([str(file_path), total_frames, f"{total_time:.4f}", f"{avg_fps:.2f}"])