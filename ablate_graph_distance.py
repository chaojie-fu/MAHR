import os
from copy import deepcopy

import numpy as np
import json
import pathlib

from config.RobotConfig.FourierGR1Config import Fourier_GR1_WholeBody_Config


INDEX_MAPPING = {
            'hipRoll_Left': 0,
            'hipRoll_Right': 6,
            'waistYaw': 12,
            'hipYaw_Left': 1,
            'hipYaw_Right': 7,
            'waistPitch': 13,
            'hipPitch_Left': 2,
            'hipPitch_Right': 8,
            'waistRoll': 14,
            'kneePitch_Left': 3,
            'kneePitch_Right': 9,
            'anklePitch_Left': 4,
            'anklePitch_Right': 10,
            'shoulderPitch_Right': 19,
            'shoulderPitch_Left': 15,
            'shoulderRoll_Right': 20,
            'ankleRoll_Right': 11,
            'ankleRoll_Left': 5,
            'shoulderRoll_Left': 16,
            'shoulderYaw_Left': 17,
            'shoulderYaw_Right': 21,
            'elbow_Left': 18,
            'elbow_Right': 22
        }


def convert_numpy_source_data_to_v2_format(npz_file, output_dir=None):
    data = np.load(npz_file)
    robot_gc = data["OriginState"][:, 0:30]
    ref_gc = data["ReferenceState"][:, 0:30]

    # in v2 json format, quaternion is (x, y, z, w)
    # while in gc, quaternion is (w, x, y, z)

    robot_v2 = {
        "fps": 100.0,
        "trajectory": []
    }

    reference_v2 = {
        "fps": 100.0,
        "trajectory": []
    }

    for frame in range(robot_gc.shape[0]):
        frame_data = {
            "RootPos": [],
            "RootOrient": [],
            "JointPos": {}
        }

        for idx in range(3):
            frame_data["RootPos"].append(float(robot_gc[frame, idx]))

        frame_data["RootOrient"].append(float(robot_gc[frame, 4]))
        frame_data["RootOrient"].append(float(robot_gc[frame, 5]))
        frame_data["RootOrient"].append(float(robot_gc[frame, 6]))
        frame_data["RootOrient"].append(float(robot_gc[frame, 3]))

        for joint_name, joint_idx in INDEX_MAPPING.items():
            frame_data["JointPos"][joint_name] = float(robot_gc[frame, 7 + joint_idx])

        robot_v2["trajectory"].append(frame_data)

    for frame in range(ref_gc.shape[0]):
        frame_data = {
            "RootPos": [],
            "RootOrient": [],
            "JointPos": {}
        }

        for idx in range(3):
            frame_data["RootPos"].append(float(ref_gc[frame, idx]))

        frame_data["RootOrient"].append(float(ref_gc[frame, 4]))
        frame_data["RootOrient"].append(float(ref_gc[frame, 5]))
        frame_data["RootOrient"].append(float(ref_gc[frame, 6]))
        frame_data["RootOrient"].append(float(ref_gc[frame, 3]))

        for joint_name, joint_idx in INDEX_MAPPING.items():
            frame_data["JointPos"][joint_name] = float(ref_gc[frame, 7 + joint_idx])

        reference_v2["trajectory"].append(frame_data)

    if output_dir is not None:
        robot_output_file = f"{output_dir}/{pathlib.Path(npz_file).stem}_robot_v2.json"
        with open(robot_output_file, "w") as f:
            json.dump(robot_v2, f, indent=2)

        reference_output_file = f"{output_dir}/{pathlib.Path(npz_file).stem}_reference_v2.json"
        with open(reference_output_file, "w") as f:
            json.dump(reference_v2, f, indent=2)

    return robot_v2, reference_v2


if __name__ == "__main__":
    root_dir = "/home/cjfu/Retarget_SOTA/MAHR/imitation_ablation"
    os.makedirs(os.path.join(root_dir, "v2_json"), exist_ok=True)
    ablation_pairs = [
        ("13_29_wo_GD.npz", "13_29_w_GD.npz"),
        ("14_03_wo_GD.npz", "14_03_w_GD.npz"),
        ("140_01_wo_GD.npz", "140_01_w_GD.npz"),
    ]
    for pair in ablation_pairs:
        print("Generating v2 json for pair:", pair)
        print("Processing:", pair[0])
        robot_json_wo, reference_json_wo = convert_numpy_source_data_to_v2_format(
            os.path.join(root_dir, pair[0]),
            output_dir=os.path.join(root_dir, "v2_json")
        )
        print("Processing:", pair[0])
        robot_json_w, reference_json_w = convert_numpy_source_data_to_v2_format(
            os.path.join(root_dir, pair[1]),
            output_dir=os.path.join(root_dir, "v2_json")
        )

        robot_config = Fourier_GR1_WholeBody_Config()
        robot = robot_config.robot
        key_joints = robot_config.key_joints

        wo_metric = {}
        w_metric = {}
        metrics = ["MPJPE", "MPJE"]
        for metric in metrics:
            wo_metric[metric] = []
            w_metric[metric] = []

        for frame in range(len(robot_json_wo["trajectory"])):
            ref_fk = deepcopy(robot.link_fk(cfg=reference_json_wo["trajectory"][frame]["JointPos"]))
            wo_fk = deepcopy(robot.link_fk(cfg=robot_json_wo["trajectory"][frame]["JointPos"]))
            w_fk = deepcopy(robot.link_fk(cfg=robot_json_w["trajectory"][frame]["JointPos"]))

            ref_j = list(reference_json_wo["trajectory"][frame]["JointPos"].values())
            wo_j = list(robot_json_wo["trajectory"][frame]["JointPos"].values())
            w_j = list(robot_json_w["trajectory"][frame]["JointPos"].values())

            for key_joint in key_joints:
                if key_joint == "ground":
                    continue
                ref_pos = np.asarray(ref_fk[key_joint][0:3, 3])
                wo_pos = np.asarray(wo_fk[key_joint][0:3, 3])
                w_pos = np.asarray(w_fk[key_joint][0:3, 3])

                wo_metric["MPJPE"].append(np.linalg.norm(wo_pos - ref_pos))
                w_metric["MPJPE"].append(np.linalg.norm(w_pos - ref_pos))

            ref_j = np.asarray(ref_j)
            wo_j = np.asarray(wo_j)
            w_j = np.asarray(w_j)

            wo_metric["MPJE"].append(np.linalg.norm(wo_j - ref_j))
            w_metric["MPJE"].append(np.linalg.norm(w_j - ref_j))


        print(f"Results for pair: {pair}")
        print(f"Without GD - MPJPE: {np.mean(wo_metric['MPJPE']):.4f} ± {np.std(wo_metric['MPJPE']):.4f}")
        print(f"With GD    - MPJPE: {np.mean(w_metric['MPJPE']):.4f} ± {np.std(w_metric['MPJPE']):.4f}")
        print(f"Without GD - MPJE: {np.mean(wo_metric['MPJE']):.4f} ± {np.std(wo_metric['MPJE']):.4f}")
        print(f"With GD    - MPJE: {np.mean(w_metric['MPJE']):.4f} ± {np.std(w_metric['MPJE']):.4f}")
        print("--------------------------------------------------")




