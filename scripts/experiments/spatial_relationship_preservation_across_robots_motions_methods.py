import argparse
import sys
import os
import pandas as pd
import sqlite3
from datetime import datetime
from mahr.utils.graph_distance.mocap_graph_generator import generate_vector_graphs_for_motion
from mahr.utils.graph_distance.robot_graph_generator import generate_vector_symbolic_graph_for_robot
from mahr.utils.cache_utils import atomic_pickle_dump, safe_pickle_load
from pathlib import Path
import json
import numpy as np
from scipy.stats import spearmanr
from mahr.config import (
    agibot_x1_urdf,
    berkeley_humanoid_lite_urdf,
    boston_dynamics_atlas_urdf,
    fourier_gr1_urdf,
    fourier_n1_urdf,
    icub_urdf,
    unitree_g1_urdf,
    unitree_h1_urdf,
)

from config.MocapConfig.AmassSmplHGConfig import AMASS_DanceDB_SMPL_HG_WholeBody_Config
from config.RobotConfig.AgibotX1Config import Agibot_X1_WholeBody_Config
from config.RobotConfig.BerkeleyHumanoidConfig import Berkeley_Humanoid_Lite_WholeBody_Config
from config.RobotConfig.BostonDynamicsAtlasHDConfig import Boston_Dynamics_Atlas_HD_WholeBody_Config
from config.RobotConfig.FourierGR1Config import Fourier_GR1_WholeBody_Config
from config.RobotConfig.FourierN1Config import Fourier_N1_WholeBody_Config
from config.RobotConfig.iCubConfig import iCub_WholeBody_Config
from config.RobotConfig.UnitreeG1 import Unitree_G1_WholeBody_Config
from config.RobotConfig.UnitreeH1 import Unitree_H1_WholeBody_Config


def get_absolute_path(relative_path):
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, relative_path)


class MotionsToCompare:
    def __init__(self):
        self.motions = [
            "144_32_poses",
            "80_43_poses",
            "82_05_poses",
            "90_28_poses",
            "A10-lie_to_crouch_poses",
            "Extended_1_poses",
            "Extended_3_poses",
            "G4-spinning_back_kick_poses",
            "G7-capoera_poses",
            "Stefanos_Zeibekiko_CY_C3D_poses"
        ]
        self.motion_file_mapping = {
            "144_32_poses": get_absolute_path("mahr/rsc/amass_bvh/CMU/144_32_poses.bvh"),
            "80_43_poses": get_absolute_path("mahr/rsc/amass_bvh/CMU/80_43_poses.bvh"),
            "82_05_poses": get_absolute_path("mahr/rsc/amass_bvh/CMU/82_05_poses.bvh"),
            "90_28_poses": get_absolute_path("mahr/rsc/amass_bvh/CMU/90_28_poses.bvh"),
            "A10-lie_to_crouch_poses": get_absolute_path("mahr/rsc/amass_bvh/ACCAD/A10-lie_to_crouch_poses.bvh"),
            "Extended_1_poses": get_absolute_path("mahr/rsc/amass_bvh/ACCAD/Extended_1_poses.bvh"),
            "Extended_3_poses": get_absolute_path("mahr/rsc/amass_bvh/ACCAD/Extended_3_poses.bvh"),
            "G4-spinning_back_kick_poses": get_absolute_path("mahr/rsc/amass_bvh/ACCAD/G4-spinning_back_kick_poses.bvh"),
            "G7-capoera_poses": get_absolute_path("mahr/rsc/amass_bvh/ACCAD/G7-capoera_poses.bvh"),
            "Stefanos_Zeibekiko_CY_C3D_poses": get_absolute_path("mahr/rsc/amass_bvh/DanceDB_SMPLHG/Stefanos_Zeibekiko_CY_C3D_poses.bvh")
        }

    def __iter__(self):
        return iter(self.motions)

    def __getitem__(self, name):
        assert name in self.motions, f"Motion {name} not found in motions to compare."
        return self.motion_file_mapping[name]


class MethodsToCompare():
    def __init__(self):
        self.methods = [
            "mahr",
            "gmr",
            "phc"
        ]

    def __iter__(self):
        return iter(self.methods)

    def joint_mapping(self, method):
        """
        return mapping from robot.actuated_joint_names to joint names in json_v2 file
        """
        if method == "mahr":
            return None
        elif method == "gmr":
            return None
        elif method == "phc":
            return None
        elif method == "ik":
            return None
        elif method == "jm":
            return None
        else:
            raise ValueError(f"Unknown method {method}.")



class RobotsToCompare:
    def __init__(self):
        self.robots = [
            "AgibotX1",
            "BostonDynamicsAtlasHD",
            "FourierN1",
            "UnitreeH1",
            "BerkeleyHumanoid",
            "FourierGR1",
            "iCub",
            "UnitreeG1",
        ]
        self.robot_config_mapping = {
            "AgibotX1": Agibot_X1_WholeBody_Config,
            "BostonDynamicsAtlasHD": Boston_Dynamics_Atlas_HD_WholeBody_Config,
            "FourierN1": Fourier_N1_WholeBody_Config,
            "UnitreeH1": Unitree_H1_WholeBody_Config,
            "BerkeleyHumanoid": Berkeley_Humanoid_Lite_WholeBody_Config,
            "FourierGR1": Fourier_GR1_WholeBody_Config,
            "iCub": iCub_WholeBody_Config,
            "UnitreeG1": Unitree_G1_WholeBody_Config,
        }

    def __iter__(self):
        return iter(self.robots)

    def __getitem__(self, name):
        assert name in self.robots, f"Robot {name} not found in motions to compare."
        return self.robot_config_mapping[name]


ROBOT_URDF_PATHS: dict[str, str] = {
    "AgibotX1": agibot_x1_urdf,
    "BostonDynamicsAtlasHD": boston_dynamics_atlas_urdf,
    "FourierN1": fourier_n1_urdf,
    "UnitreeH1": unitree_h1_urdf,
    "BerkeleyHumanoid": berkeley_humanoid_lite_urdf,
    "FourierGR1": fourier_gr1_urdf,
    "iCub": icub_urdf,
    "UnitreeG1": unitree_g1_urdf,
}


METHOD_NAME_MAPPING = {
    "mahr": "GD(Ours)",
    "gmr": "GMR",
    "phc": "PHC",
    "ik": "IK",
    "jm": "JM",
}

PAPER_METHOD_ORDER = ["GMR", "PHC", "JM", "IK", "GD(Ours)"]
PAPER_FONT_FAMILY = "Helvetica"


def prepare_method_label_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw method names to publication labels with stable plotting order."""
    labeled_df = df.copy()
    labeled_df["Methods"] = labeled_df["method_name"].map(METHOD_NAME_MAPPING)
    labeled_df = labeled_df[labeled_df["Methods"].notna()].copy()
    labeled_df["Methods"] = pd.Categorical(
        labeled_df["Methods"],
        categories=PAPER_METHOD_ORDER,
        ordered=True,
    )
    return labeled_df


# -----------------------------
# Database helpers
# -----------------------------

def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def open_db(db_path: str) -> sqlite3.Connection:
    """Open a single connection configured for concurrent writers.

    - WAL mode for better concurrency
    - synchronous=NORMAL for speed with durability tradeoff acceptable here
    - busy_timeout to wait for locks instead of failing
    """
    _ensure_parent_dir(db_path)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn


UPSERT_SQL = (
    """
    INSERT INTO spearman_results (
        motion_name, robot_name, method_name, frame_index, value, motion_file
    ) VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(motion_name, robot_name, method_name, frame_index)
    DO UPDATE SET
        value=excluded.value,
        created_at=datetime('now'),
        motion_file=excluded.motion_file
    """
)


def batch_upsert_spearman(conn: sqlite3.Connection, rows: list[tuple]):
    """Batch insert/update rows in a single transaction."""
    if not rows:
        return 0
    cur = conn.cursor()
    cur.execute("BEGIN")
    cur.executemany(UPSERT_SQL, rows)
    conn.commit()
    return len(rows)


# -----------------------------
# Caching helpers
# -----------------------------

def cached_motion_config(cache_dir, bvh_motion_file, norm_pose_dir, motion_name):
    """
    Load or build and cache a motion config for a given motion file.

    Cache key encodes motion name, normalization pose, and motion_range.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = Path(cache_dir) / f"{motion_name}.pkl"

    if cache_file.exists():
        cfg = safe_pickle_load(str(cache_file))
        if cfg is not None:
            return cfg
        # Fallthrough if cache invalid

    _motion_config = AMASS_DanceDB_SMPL_HG_WholeBody_Config()
    _motion_config.load_motion(bvh_motion_file, norm_pose_dir=norm_pose_dir)
    atomic_pickle_dump(_motion_config, str(cache_file))
    return _motion_config


def load_json_v2(json_file, robot_joint_names, joint_name_mapping=None):
    with open(json_file, 'r') as f:
        data = json.load(f)

    fps = data['fps']
    num_frames = len(data['trajectory'])
    num_joints = len(data['trajectory'][0]['JointPos'])
    root_pos = np.zeros([num_frames, 3], dtype=float)
    root_orient = np.zeros([num_frames, 4], dtype=float)
    joint_pos = np.zeros([num_frames, num_joints], dtype=float)

    for frame_idx, frame in enumerate(data['trajectory']):
        for idx in range(3):
            root_pos[frame_idx, idx] = frame['RootPos'][idx]
        for idx in range(4): # xyzw
            root_orient[frame_idx, idx] = frame['RootOrient'][idx]
        for joint_idx, joint_name in enumerate(robot_joint_names):
            if joint_name_mapping is not None:
                mapped_name = joint_name_mapping[joint_name]
                joint_pos[frame_idx, joint_idx] = frame['JointPos'][mapped_name]
            else:
                joint_pos[frame_idx, joint_idx] = frame['JointPos'][joint_name]

    return fps, num_frames, root_pos, root_orient, joint_pos


def quat2rotmat(quat):
    """
    Convert quaternion (xyzw) to rotation matrix.
    """
    x, y, z, w = quat
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ], dtype=float)
    return R

def get_json_file(motion_name, robot_name, method_name):
    root_dir = "retargeted_motions"
    method_dir_name = None
    robot_dir_name = None
    motion_file_name = None
    match method_name:
        case "mahr":
            method_dir_name = f"MAHR_Wider_Constraint"
            match robot_name:
                case "AgibotX1":
                    robot_dir_name = "agibot_x1"
                case "BostonDynamicsAtlasHD":
                    robot_dir_name = "boston_dynamics_atlas_hd"
                case "FourierN1":
                    robot_dir_name = "fourier_n1"
                case "UnitreeH1":
                    robot_dir_name = "unitree_h1"
                case "BerkeleyHumanoid":
                    robot_dir_name = "berkeley_humanoid_lite"
                case "FourierGR1":
                    robot_dir_name = "fourier_gr1"
                case "iCub":
                    robot_dir_name = "icub"
                case "UnitreeG1":
                    robot_dir_name = "unitree_g1"
                case _:
                    raise ValueError(f"Unknown robot name: {robot_name}")
            match motion_name:
                case "144_32_poses":
                    motion_file_name = "144_32_poses_gd_v2.json"
                case "80_43_poses":
                    motion_file_name = "80_43_poses_gd_v2.json"
                case "82_05_poses":
                    motion_file_name = "82_05_poses_gd_v2.json"
                case "90_28_poses":
                    motion_file_name = "90_28_poses_gd_v2.json"
                case "A10-lie_to_crouch_poses":
                    motion_file_name = "A10-lie_to_crouch_poses_gd_v2.json"
                case "Extended_1_poses":
                    motion_file_name = "Extended_1_poses_gd_v2.json"
                case "Extended_3_poses":
                    motion_file_name = "Extended_3_poses_gd_v2.json"
                case "G4-spinning_back_kick_poses":
                    motion_file_name = "G4-spinning_back_kick_poses_gd_v2.json"
                case "G7-capoera_poses":
                    motion_file_name = "G7-capoera_poses_gd_v2.json"
                case "Stefanos_Zeibekiko_CY_C3D_poses":
                    motion_file_name = "Stefanos_Zeibekiko_CY_C3D_poses_gd_v2.json"
                case _:
                    raise ValueError(f"Unknown motion name: {motion_name}")

        case "gmr":
            method_dir_name = f"GMR"
            match robot_name:
                case "BerkeleyHumanoid":
                    robot_dir_name = "berkeley_humanoid_lite"
                case "FourierN1":
                    robot_dir_name = "fourier_n1"
                case "UnitreeG1":
                    robot_dir_name = "unitree_g1"
                case "UnitreeH1":
                    robot_dir_name = "unitree_h1"
                case _:
                    raise ValueError(f"Unknown robot name: {robot_name}")
            match motion_name:
                case "144_32_poses":
                    motion_file_name = "144_32_stageii.npz.json"
                case "80_43_poses":
                    motion_file_name = "80_43_stageii.npz.json"
                case "82_05_poses":
                    motion_file_name = "82_05_stageii.npz.json"
                case "90_28_poses":
                    motion_file_name = "90_28_stageii.npz.json"
                case "A10-lie_to_crouch_poses":
                    motion_file_name = "A10_-_lie_to_crouch_stageii.npz.json"
                case "Extended_1_poses":
                    motion_file_name = "Extended_1_stageii.npz.json"
                case "Extended_3_poses":
                    motion_file_name = "Extended_3_stageii.npz.json"
                case "G4-spinning_back_kick_poses":
                    motion_file_name = "G4_-spinning_back_kick_stageii.npz.json"
                case "G7-capoera_poses":
                    motion_file_name = "G7_-__capoera_stageii.npz.json"
                case "Stefanos_Zeibekiko_CY_C3D_poses":
                    motion_file_name = "Stefanos_Zeibekiko_CY_C3D_stageii.npz.json"
                case _:
                    raise ValueError(f"Unknown motion name: {motion_name}")

        case "phc":
            method_dir_name = "PHC"
            match robot_name:
                case "UnitreeH1":
                    robot_dir_name = "Unitree_H1_PROCESSED"
                case "UnitreeG1":
                    robot_dir_name = "Unitree_G1_PROCESSED"
                case _:
                    raise ValueError(f"Unknown robot name: {robot_name}")
            match motion_name:
                case "144_32_poses":
                    motion_file_name = "0-selected_data_144_32_stageii.json"
                case "80_43_poses":
                    motion_file_name = "0-selected_data_80_43_stageii.json"
                case "82_05_poses":
                    motion_file_name = "0-selected_data_82_05_stageii.json"
                case "90_28_poses":
                    motion_file_name = "0-selected_data_90_28_stageii.json"
                case "A10-lie_to_crouch_poses":
                    motion_file_name = "0-selected_data_A10_-_lie_to_crouch_stageii.json"
                case "Extended_1_poses":
                    motion_file_name = "0-selected_data_Extended_1_stageii.json"
                case "Extended_3_poses":
                    motion_file_name = "0-selected_data_Extended_3_stageii.json"
                case "G4-spinning_back_kick_poses":
                    motion_file_name = "0-selected_data_G4_-spinning_back_kick_stageii.json"
                case "G7-capoera_poses":
                    motion_file_name = "0-selected_data_G7_-__capoera_stageii.json"
                case "Stefanos_Zeibekiko_CY_C3D_poses":
                    motion_file_name = "0-selected_data_Stefanos_Zeibekiko_CY_C3D_stageii.json"
                case _:
                    raise ValueError(f"Unknown motion name: {motion_name}")
        case "ik":
            method_dir_name = "IK"
            match robot_name:
                case "FourierGR1":
                    robot_dir_name = "GR1"
                case _:
                    raise ValueError(f"Unknown robot name: {robot_name}")
            match motion_name:
                case "80_43_poses":
                    motion_file_name = "80_43_poses_ik_v2.json"
                case "82_05_poses":
                    motion_file_name = "82_05_poses_ik_v2.json"
                case "90_28_poses":
                    motion_file_name = "90_28_poses_ik_v2.json"
                case "A10-lie_to_crouch_poses":
                    motion_file_name = "A10-lie_to_crouch_poses_ik_v2.json"
                case "Extended_1_poses":
                    motion_file_name = "Extended_1_poses_ik_v2.json"
                case "Extended_3_poses":
                    motion_file_name = "Extended_3_poses_ik_v2.json"
                case "G4-spinning_back_kick_poses":
                    motion_file_name = "G4-spinning_back_kick_poses_ik_v2.json"
                case "G7-capoera_poses":
                    motion_file_name = "G7-capoera_poses_ik_v2.json"
                case _:
                    raise ValueError(f"Unknown motion name: {motion_name}")

        case "jm":
            method_dir_name = "JM"
            match robot_name:
                case "FourierGR1":
                    robot_dir_name = "GR1"
                case _:
                    raise ValueError(f"Unknown robot name: {robot_name}")
            match motion_name:
                case "80_43_poses":
                    motion_file_name = "80_43_poses_jm_v2.json"
                case "82_05_poses":
                    motion_file_name = "82_05_poses_jm_v2.json"
                case "90_28_poses":
                    motion_file_name = "90_28_poses_jm_v2.json"
                case "A10-lie_to_crouch_poses":
                    motion_file_name = "A10-lie_to_crouch_poses_jm_v2.json"
                case "Extended_1_poses":
                    motion_file_name = "Extended_1_poses_jm_v2.json"
                case "Extended_3_poses":
                    motion_file_name = "Extended_3_poses_jm_v2.json"
                case "G4-spinning_back_kick_poses":
                    motion_file_name = "G4-spinning_back_kick_poses_jm_v2.json"
                case "G7-capoera_poses":
                    motion_file_name = "G7-capoera_poses_jm_v2.json"
                case _:
                    raise ValueError(f"Unknown motion name: {motion_name}")

    return os.path.join(root_dir, method_dir_name, robot_dir_name, motion_file_name)


def data_generation(motion_name, robot_name, method_name, db_path=None, write_db=False):
    json_file = get_json_file(motion_name, robot_name, method_name)
    motions = MotionsToCompare()
    all_norm_pose_dir = get_absolute_path("mahr/rsc/amass_bvh/norm_pose")

    cache_dirs = {
        "motion_dict": get_absolute_path("cache/motion_dict"),
        "motion_config": get_absolute_path("cache/motion_config"),
        "motion_graph": get_absolute_path("cache/motion_graphs"),
        "robot_graph": get_absolute_path("cache/robot_graphs"),
    }
    for target_dir in cache_dirs.values():
        os.makedirs(target_dir, exist_ok=True)

    # construct all motion graphs for comparison
    # first, load motion config for each motion (with caching)
    motion_bvh = motions[motion_name]
    motion_config = cached_motion_config(cache_dirs["motion_config"], motion_bvh, all_norm_pose_dir, motion_name)

    # then, compute motion graphs for each motion
    _, motion_graphs, _, _ = generate_vector_graphs_for_motion(
        motion_config,
        cache_dirs["motion_dict"],
        cache_dirs["motion_graph"],
        normalize_edge_weights=False,
        force_generate_motion_dict=False,
        force_generate_graph=False,
        filter_edges_when_generating_graph=False,
        edge_filter=None
    )

    robots = RobotsToCompare()
    # after constructing motion graphs, construct robot graphs
    # first load robot config for each robot
    robot_config = robots[robot_name]()

    # then, compute robot graphs for each robot and different retargeting methods
    _, _, _, robot_graph_callable, _ = generate_vector_symbolic_graph_for_robot(
        robot_config,
        normalize_edge_weights=False,
        filter_edges_when_generating_graph=False,
        edge_filter=None)

    robot_graphs = []

    methods = MethodsToCompare()
    method_joint_mapping = methods.joint_mapping(method_name)
    # load json_v2 file for robot
    robot_joint_names = robot_graph_callable.robot.actuated_joint_names
    fps, num_frames, root_pos, root_orient, joint_pos = load_json_v2(
        json_file,
        robot_joint_names,
        joint_name_mapping=method_joint_mapping)

    for frame in range(num_frames):
        robot_graphs.append(
            robot_graph_callable(
                joint_pos[frame, :].tolist(),
                quat2rotmat(root_orient[frame, :]),
                root_pos[frame, 0],
                root_pos[frame, 1],
                root_pos[frame, 2]
            )
        )

    # finally, save robot graphs to disk for later use
    atomic_pickle_dump(robot_graphs, os.path.join(cache_dirs["robot_graph"], f"{robot_name}.pkl"))

    # finally, compute spatial relationship preservation metrics across different robots and retargeting methods
    motion_key_bodies = motion_config.key_bodies
    robot_key_joints = robot_config.key_joints

    assert len(motion_key_bodies) == len(robot_key_joints)
    num_vertices = len(motion_key_bodies)

    # exclude ground node from edges to consider
    edge_indices = []
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            edge_indices.append((i, j))
            # if motion_key_bodies[i] != "ground" and motion_key_bodies[j] != "ground":
            #     edge_indices.append((i, j))

    num_edges = len(edge_indices)
    motion_lengths = np.zeros([num_edges])
    robot_lengths = np.zeros([num_edges])

    spearmanrs_list = []
    db_rows: list[tuple] = []
    conn: sqlite3.Connection | None = None
    if write_db and db_path:
        conn = open_db(db_path)
        # Ensure schema exists using the same connection
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS spearman_results (
                motion_name   TEXT NOT NULL,
                robot_name    TEXT NOT NULL,
                method_name   TEXT NOT NULL,
                frame_index   INTEGER NOT NULL,
                value         REAL NOT NULL,
                created_at    TEXT NOT NULL DEFAULT (datetime('now')),
                motion_file   TEXT NOT NULL,
                PRIMARY KEY (motion_name, robot_name, method_name, frame_index)
            )
            """
        )
        conn.commit()

    for frame_idx in range(min(len(motion_graphs), len(robot_graphs))):
        for edge_idx, (edge_start_idx, edge_end_idx) in enumerate(edge_indices):
            motion_ss = motion_key_bodies[edge_start_idx]
            motion_to = motion_key_bodies[edge_end_idx]
            robot_ss = robot_key_joints[edge_start_idx]
            robot_to = robot_key_joints[edge_end_idx]

            mo_pos_vec = motion_graphs[frame_idx].get_edge_data(motion_ss, motion_to)["position_vector"]
            motion_lengths[edge_idx] = np.linalg.norm(mo_pos_vec, ord=2)

            ro_pos_vec = robot_graphs[frame_idx].get_edge_data(robot_ss, robot_to)["position_vector"]
            robot_lengths[edge_idx] = np.linalg.norm(ro_pos_vec, ord=2)

        spearmanrs = spearmanr(motion_lengths, robot_lengths).statistic
        spearmanrs_list.append(spearmanrs)

        # Queue DB row; we'll batch insert later
        if conn is not None:
            db_rows.append((
                motion_name,
                robot_name,
                method_name,
                int(frame_idx),
                float(spearmanrs),
                json_file
            ))

    # Batch-write to database in a single transaction
    if conn is not None:
        written = batch_upsert_spearman(conn, db_rows)
        print(f"Wrote {written} frame rows to database at {db_path} (WAL mode).")
        conn.close()


def analysis(db_path, analysis_db_path):
    conn = open_db(db_path)
    analysis_conn = open_db(analysis_db_path)

    # construct the new analysis table, which constructs columns with motion_name, robot_name, frame_idx and multiple method_name columns, each containing the spearman value
    analysis_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS spearman_comparison(
            motion_name TEXT,
            robot_name TEXT,
            frame_idx INTEGER,
            PRIMARY KEY (motion_name, robot_name, frame_idx)
        )
        """
    )

    # get existing columns in the analysis table
    cursor = analysis_conn.execute("PRAGMA table_info(spearman_comparison);")
    columns = [row[1] for row in cursor.fetchall()]

    # add column only if missing
    for new_column in ["mahr", "gmr", "gmr_vs_mahr_diff", "phc", "phc_vs_mahr_diff", "ik", "ik_vs_mahr_diff", "jm", "jm_vs_mahr_diff"]:
        if new_column not in columns:
            analysis_conn.execute(f"ALTER TABLE `spearman_comparison` ADD COLUMN `{new_column}` REAL;")
            conn.commit()

    for new_column in ["mahr_motion_file", "gmr_motion_file", "phc_motion_file", "ik_motion_file", "jm_motion_file"]:
        if new_column not in columns:
            analysis_conn.execute(f"ALTER TABLE `spearman_comparison` ADD COLUMN `{new_column}` TEXT;")
            conn.commit()

    # query to get all unique motion_name, robot_name, frame_idx combinations
    query = """
    SELECT DISTINCT motion_name, robot_name, frame_index
    FROM spearman_results
    """
    
    # add entry to the analysis_conn for each unique motion_name, robot_name, frame_idx combination
    df_unique = pd.read_sql_query(query, conn)
    for index, row in df_unique.iterrows():
        motion_name = row['motion_name']
        robot_name = row['robot_name']
        frame_idx = row['frame_index']

        # get spearman values for each method
        query = f"""
        SELECT method_name, value, motion_file
        FROM spearman_results
        WHERE motion_name = '{motion_name}' AND robot_name = '{robot_name}' AND frame_index = {frame_idx}
        """
        df_methods = pd.read_sql_query(query, conn)
        spearman_dict = {}
        motion_file_dict = {}
        for _, method_row in df_methods.iterrows():
            spearman_dict[method_row['method_name']] = method_row['value']
            motion_file_dict[method_row['method_name']] = method_row['motion_file']

        gmr_value = spearman_dict.get('gmr', None)
        mahr_value = spearman_dict.get('mahr', None)
        phc_value = spearman_dict.get('phc', None)
        ik_value = spearman_dict.get('ik', None)
        jm_value = spearman_dict.get('jm', None)
        gmr_vs_mahr_diff = None
        phc_vs_mahr_diff = None
        ik_vs_mahr_diff = None
        jm_vs_mahr_diff = None
        if gmr_value is not None and mahr_value is not None:
            gmr_vs_mahr_diff = gmr_value - mahr_value
        if phc_value is not None and mahr_value is not None:
            phc_vs_mahr_diff = phc_value - mahr_value
        if ik_value is not None and mahr_value is not None:
            ik_vs_mahr_diff = ik_value - mahr_value
        if jm_value is not None and mahr_value is not None:
            jm_vs_mahr_diff = jm_value - mahr_value

        gmr_motion_file = motion_file_dict.get('gmr', None)
        mahr_motion_file = motion_file_dict.get('mahr', None)
        phc_motion_file = motion_file_dict.get('phc', None)
        ik_motion_file = motion_file_dict.get('ik', None)
        jm_motion_file = motion_file_dict.get('jm', None)

        analysis_conn.execute("""
            INSERT INTO spearman_comparison (motion_name, robot_name, frame_idx, 
                                             gmr, mahr, gmr_vs_mahr_diff, 
                                             phc, phc_vs_mahr_diff, 
                                             ik, ik_vs_mahr_diff, 
                                             jm, jm_vs_mahr_diff, 
                                             mahr_motion_file, 
                                             gmr_motion_file, 
                                             phc_motion_file, 
                                             ik_motion_file, 
                                             jm_motion_file)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(motion_name, robot_name, frame_idx)
            DO UPDATE SET
                gmr=excluded.gmr,
                mahr=excluded.mahr,
                gmr_vs_mahr_diff=excluded.gmr_vs_mahr_diff,
                phc=excluded.phc,
                phc_vs_mahr_diff=excluded.phc_vs_mahr_diff,
                ik=excluded.ik,
                ik_vs_mahr_diff=excluded.ik_vs_mahr_diff,
                jm=excluded.jm,
                jm_vs_mahr_diff=excluded.jm_vs_mahr_diff,
                mahr_motion_file=excluded.mahr_motion_file,
                gmr_motion_file=excluded.gmr_motion_file,
                phc_motion_file=excluded.phc_motion_file,
                ik_motion_file=excluded.ik_motion_file,
                jm_motion_file=excluded.jm_motion_file
        """, (
            motion_name, robot_name, frame_idx,
            gmr_value, mahr_value, gmr_vs_mahr_diff,
            phc_value, phc_vs_mahr_diff,
            ik_value, ik_vs_mahr_diff,
            jm_value, jm_vs_mahr_diff,
            mahr_motion_file, gmr_motion_file, phc_motion_file, ik_motion_file, jm_motion_file))

    conn.close()
    analysis_conn.commit()
    analysis_conn.close()

def plot(db_path, analysis_db_path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import ptitprince as pt

    # Helper to control fonts for RainCloud axes
    def _tune_raincloud_fonts(ax, *, title_size=None, label_size=None, tick_label_size=None, legend_size=None):
        # title
        if title_size is not None:
            try:
                ax.title.set_fontsize(title_size)
            except Exception:
                ax.set_title(ax.get_title(), fontsize=title_size)
        # axis labels
        if label_size is not None:
            ax.xaxis.label.set_size(label_size)
            ax.yaxis.label.set_size(label_size)
        # tick labels
        if tick_label_size is not None:
            ax.tick_params(axis='both', which='both', labelsize=tick_label_size)
        # legend
        leg = ax.get_legend()
        if leg is not None and legend_size is not None:
            try:
                leg.get_title().set_fontsize(legend_size)
            except Exception:
                pass
            for text in leg.get_texts():
                text.set_fontsize(legend_size)

    # Global switches
    plot_figure_1 = True
    plot_figure_2 = True
    plot_paper_boxplot_figure = True

    # Global font control (increase or decrease overall sizes here)
    font_scale = 1.2  # adjust this scalar for global scaling
    sns.set_context("paper", font_scale=font_scale)

    # Specific font sizes (override per element if desired)
    title_size = 20
    label_size = 18
    tick_label_size = 16
    legend_size = 16


    source_conn = open_db(db_path)
    analysis_conn = open_db(analysis_db_path)
    os.makedirs("results", exist_ok=True)

    # ======================================================================
    # ==== GMR vs PHC vs MAHR Spearman's Coefficient Difference Across Robots =====
    if plot_figure_1:
        print("Plotting GMR vs MAHR Spearman's Coefficient Difference Across Robots...")
        query = """
        SELECT robot_name, method_name, value
        FROM spearman_results 
        """
        df = pd.read_sql_query(query, source_conn)
        # replace robot_name to more readable format
        robot_name_mapping = {
            "AgibotX1": "Agibot\nX1",
            "BostonDynamicsAtlasHD": "Boston Dynamics\nAtlas HD",
            "FourierN1": "Fourier\nN1",
            "UnitreeH1": "Unitree\nH1",
            "BerkeleyHumanoid": "Berkeley\nHumanoid Lite",
            "FourierGR1": "Fourier\nGR1",
            "iCub": "iCub",
            "UnitreeG1": "Unitree\nG1",
        }
        df['robot_name'] = df['robot_name'].map(robot_name_mapping)

        method_name_mapping = {
            "mahr": "GD(Ours)",
            "gmr": "GMR",
            "phc": "PHC",
            "ik": "IK",
            "jm": "JM",
        }
        df['method_name'] = df['method_name'].map(method_name_mapping)

        # replace column name of method_name to Methods
        df = df.rename(columns={
            "method_name": "Methods",
            "value": "Spearman's Coefficient",
            "robot_name": "Robots"
        })


        f, ax = plt.subplots(figsize=(8, 12))
        ax = pt.RainCloud(
            x="Robots",
            y="Spearman's Coefficient",
            hue="Methods",
            order=[
                "Fourier\nGR1",
                "Unitree\nG1",
                "Unitree\nH1",
                "Fourier\nN1",
                "Berkeley\nHumanoid Lite",
                "Agibot\nX1",
                "Boston Dynamics\nAtlas HD",
                "iCub",
            ],
            hue_order=["GD(Ours)", "GMR", "PHC", "IK", "JM"],
            data=df,
            palette="Set2",
            bw=0.1,
            width_viol=0.9,
            width_box=0.40,
            ax=ax,
            orient="h",
            alpha=0.65,
            dodge=True,
            pointplot=False,
            point_size=0,
            jitter=True,
            offset=0.25,
            move=0.2,
            box_showfliers=False,
        )

        ax.set_ylabel("")

        ax.set_xlim(0.80, 1.0)
        ax.set_title("Spatial Relationship Preservation\nAcross Robots and Methods")

        # Apply font tuning for this plot
        _tune_raincloud_fonts(
            ax,
            title_size=title_size,
            label_size=label_size,
            tick_label_size=tick_label_size,
            legend_size=legend_size,
        )

        f.set_tight_layout(True)
        f.savefig("results/gmr_phc_vs_mahr_spearman_across_robots.png", dpi=300)
        plt.show()
    # ======================================================================

    def new_line_for_long_labels(label, max_char=10):
        if len(label) <= max_char:
            return label
        else:
            # break the label into multiple lines only after spaces
            parts = []
            current_part = []
            for word in label.split(" "):
                if len(" ".join(current_part + [word])) <= max_char:
                    current_part.append(word)
                else:
                    parts.append(" ".join(current_part))
                    current_part = [word]
            parts.append(" ".join(current_part))
            return '\n'.join(parts)

    def shorten_motion_name(name):
        match name:
            case "144_32_poses":
                return new_line_for_long_labels("sun salutation")
            case "80_43_poses":
                return new_line_for_long_labels("happy")
            case "82_05_poses":
                return new_line_for_long_labels("sitting on ground relaxing")
            case "90_28_poses":
                return new_line_for_long_labels("breakdance")
            case "A10-lie_to_crouch_poses":
                return new_line_for_long_labels("lie to crouch")
            case "G4-spinning_back_kick_poses":
                return new_line_for_long_labels("spinning back kick")
            case "G7-capoera_poses":
                return new_line_for_long_labels("capoera")
            case "Extended_1_poses":
                return new_line_for_long_labels("extended 1")
            case "Extended_3_poses":
                return new_line_for_long_labels("extended 3")
            case "Stefanos_Zeibekiko_CY_C3D_poses":
                return new_line_for_long_labels("zeibekiko")
            case _:
                raise ValueError(f"Unknown motion name: {name}")

    def _plot_paper_boxplot_figure():
        print("Plotting paper boxplot figure for spatial relationship preservation...")
        robot_name_mapping = {
            "AgibotX1": "Agibot\nX1",
            "BostonDynamicsAtlasHD": "Boston Dynamics\nAtlas HD",
            "FourierN1": "Fourier\nN1",
            "UnitreeH1": "Unitree\nH1",
            "BerkeleyHumanoid": "Berkeley\nHumanoid Lite",
            "FourierGR1": "Fourier\nGR1",
            "iCub": "iCub",
            "UnitreeG1": "Unitree\nG1",
        }
        robot_order = [
            "Fourier\nGR1",
            "Unitree\nG1",
            "Unitree\nH1",
            "Fourier\nN1",
            "Berkeley\nHumanoid Lite",
            "Agibot\nX1",
            "Boston Dynamics\nAtlas HD",
            "iCub",
        ]
        palette = {
            "GMR": "#4C78A8",
            "PHC": "#F58518",
            "JM": "#54A24B",
            "IK": "#B279A2",
            "GD(Ours)": "#E45756",
        }

        unitree_query = """
        SELECT motion_name, method_name, value
        FROM spearman_results
        WHERE robot_name = 'UnitreeG1'
        """
        unitree_df = pd.read_sql_query(unitree_query, source_conn)
        unitree_df = prepare_method_label_dataframe(unitree_df)
        unitree_df["Motions"] = unitree_df["motion_name"].map(shorten_motion_name)
        motion_order = sorted(unitree_df["Motions"].dropna().unique().tolist())
        unitree_df = unitree_df.rename(columns={"value": "Spearman's Coefficient"})

        robots_query = """
        SELECT robot_name, method_name, value
        FROM spearman_results
        """
        robots_df = pd.read_sql_query(robots_query, source_conn)
        robots_df = prepare_method_label_dataframe(robots_df)
        robots_df["Robots"] = robots_df["robot_name"].map(robot_name_mapping)
        robots_df = robots_df.rename(columns={"value": "Spearman's Coefficient"})

        rc_params = {
            "font.family": "sans-serif",
            "font.sans-serif": [PAPER_FONT_FAMILY, "Arial", "DejaVu Sans"],
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.6,
            "axes.labelsize": 7.5,
            "axes.titlesize": 7.8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }

        with plt.rc_context(rc_params):
            sns.set_theme(
                context="paper",
                style="whitegrid",
                font=PAPER_FONT_FAMILY,
                rc=rc_params,
            )
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(7.16, 3.95),
                sharex=False,
                constrained_layout=False,
            )
            fig.subplots_adjust(left=0.095, right=0.965, bottom=0.22, top=0.88, wspace=0.42)

            common_boxplot_kwargs = {
                "hue": "Methods",
                "hue_order": PAPER_METHOD_ORDER,
                "palette": palette,
                "orient": "h",
                "showfliers": False,
                "linewidth": 0.55,
                "width": 0.68,
                "dodge": True,
                "saturation": 0.75,
            }

            def _add_alternating_group_bands(ax, num_groups: int):
                for group_idx in range(num_groups):
                    if group_idx % 2 == 0:
                        ax.axhspan(
                            group_idx - 0.5,
                            group_idx + 0.5,
                            facecolor="#ECECEC",
                            edgecolor="none",
                            zorder=0,
                        )
                    ax.axhline(
                        group_idx + 0.5,
                        color="#D8D8D8",
                        linewidth=0.3,
                        zorder=0.5,
                    )

            def _add_median_markers(ax, df: pd.DataFrame, group_col: str, group_order: list[str]):
                box_height = common_boxplot_kwargs["width"] / len(PAPER_METHOD_ORDER)
                offsets = np.linspace(
                    -common_boxplot_kwargs["width"] / 2 + box_height / 2,
                    common_boxplot_kwargs["width"] / 2 - box_height / 2,
                    len(PAPER_METHOD_ORDER),
                )
                for group_idx, group_name in enumerate(group_order):
                    group_df = df[df[group_col] == group_name]
                    for method_idx, method_name in enumerate(PAPER_METHOD_ORDER):
                        method_values = group_df.loc[
                            group_df["Methods"] == method_name,
                            "Spearman's Coefficient",
                        ]
                        if method_values.empty:
                            continue
                        ax.scatter(
                            method_values.median(),
                            group_idx + offsets[method_idx],
                            s=8,
                            marker="o",
                            facecolor="white",
                            edgecolor="#222222",
                            linewidth=0.35,
                            zorder=4,
                        )

            sns.boxplot(
                data=unitree_df,
                x="Spearman's Coefficient",
                y="Motions",
                order=motion_order,
                ax=axes[0],
                **common_boxplot_kwargs,
            )
            _add_alternating_group_bands(axes[0], len(motion_order))
            _add_median_markers(axes[0], unitree_df, "Motions", motion_order)
            axes[0].set_title("Unitree G1 Spatial Relationship Preservation\nAcross Motions and Methods", loc="center", fontweight="normal")
            axes[0].set_xlabel("Spearman's Coefficient (higher is better)")
            axes[0].set_ylabel("")
            axes[0].set_xlim(0.70, 1.00)

            sns.boxplot(
                data=robots_df,
                x="Spearman's Coefficient",
                y="Robots",
                order=robot_order,
                ax=axes[1],
                **common_boxplot_kwargs,
            )
            _add_alternating_group_bands(axes[1], len(robot_order))
            _add_median_markers(axes[1], robots_df, "Robots", robot_order)
            axes[1].set_title("Spatial Relationship Preservation\nAcross Robots and Methods", loc="center", fontweight="normal")
            axes[1].set_xlabel("Spearman's Coefficient (higher is better)")
            axes[1].set_ylabel("")
            axes[1].set_xlim(0.80, 1.00)
            axes[1].set_xticks(np.arange(0.80, 1.001, 0.05))

            for ax, panel_label in zip(axes, ["a.", "b."]):
                ax.text(
                    -0.14,
                    1.085,
                    panel_label,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontweight="normal",
                    fontsize=9,
                )

            legend_handles, legend_labels = axes[1].get_legend_handles_labels()
            ordered_handles = []
            ordered_labels = []
            for method_name in PAPER_METHOD_ORDER:
                if method_name in legend_labels:
                    ordered_handles.append(legend_handles[legend_labels.index(method_name)])
                    ordered_labels.append(method_name)
            for ax in axes:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
                ax.grid(True, axis="x", color="#D9D9D9", linewidth=0.35)
                ax.grid(False, axis="y")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(axis="both", length=2.2, width=0.55, color="#222222")

            legend = fig.legend(
                ordered_handles,
                ordered_labels,
                title="Methods",
                loc="lower center",
                ncol=len(PAPER_METHOD_ORDER),
                frameon=True,
                fancybox=False,
                framealpha=1.0,
                edgecolor="#222222",
                bbox_to_anchor=(0.5, 0.018),
            )
            legend.get_frame().set_linewidth(0.5)
            fig.savefig("results/spatial_relationship_preservation_boxplot_paper.png", dpi=600)
            fig.savefig("results/spatial_relationship_preservation_boxplot_paper.pdf")
            plt.close(fig)

    if plot_paper_boxplot_figure:
        _plot_paper_boxplot_figure()

    # ======================================================================
    # ==== GMR vs MAHR Spearman's Coefficient Difference Across Motions =====
    # choose UnitreeG1 robot for analysis
    if plot_figure_2:
        print("Plotting GMR, PHC and MAHR Spearman's Coefficient Difference Across Motions On Unitree G1...")
        for target_robot in ["UnitreeG1", "UnitreeH1", "FourierGR1"]:
            robot_name_mapping = {
                "UnitreeG1": "Unitree G1",
                "UnitreeH1": "Unitree H1",
                "FourierGR1": "Fourier GR1",
            }
            scale_mapping = {
                "UnitreeG1": [0.7, 1.0],
                "UnitreeH1": [0.5, 1.0],
                "FourierGR1": [0.6, 1.0],
            }
            query = f"""
            SELECT motion_name, method_name, value
            FROM spearman_results 
            WHERE robot_name = '{target_robot}'
            """
            df = pd.read_sql_query(query, source_conn)

            unique_motions = df['motion_name'].unique()

            motion_name_mapping = {
                unique_motion_name: shorten_motion_name(unique_motion_name) for unique_motion_name in unique_motions
            }
            df["motion_name"] = df["motion_name"].map(motion_name_mapping)

            # get all unique motion_names from df and save them into a list
            unique_sorted_motion_names = sorted(df['motion_name'].unique().tolist())

            method_name_mapping = {
                "mahr": "GD(Ours)",
                "gmr": "GMR",
                "phc": "PHC",
                "ik": "IK",
                "jm": "JM",
            }
            df['method_name'] = df['method_name'].map(method_name_mapping)

            # replace column name of method_name to Methods
            df = df.rename(columns={
                "method_name": "Methods",
                "value": "Spearman's Coefficient",
                "motion_name": "Motions"
            })

            f, ax = plt.subplots(figsize=(8, 12))
            if target_robot in ["FourierGR1"]:
                ax = pt.RainCloud(
                    x="Motions",
                    y="Spearman's Coefficient",
                    order=unique_sorted_motion_names,
                    hue="Methods",
                    hue_order=["GD(Ours)", "IK", "JM"],
                    data=df,
                    palette="Set2",
                    bw=0.1,
                    width_viol=0.9,
                    width_box=0.40,
                    ax=ax,
                    orient="h",
                    alpha=0.65,
                    dodge=True,
                    pointplot=False,
                    point_size=0,
                    jitter=True,
                    offset=0.25,
                    move=0.2,
                    box_showfliers=False,
                )
            elif ["UnitreeG1", "UnitreeH1"]:
                ax = pt.RainCloud(
                    x="Motions",
                    y="Spearman's Coefficient",
                    order=unique_sorted_motion_names,
                    hue="Methods",
                    hue_order=["GD(Ours)", "GMR", "PHC"],
                    data=df,
                    palette="Set2",
                    bw=0.1,
                    width_viol=0.9,
                    width_box=0.40,
                    ax=ax,
                    orient="h",
                    alpha=0.65,
                    dodge=True,
                    pointplot=False,
                    point_size=0,
                    jitter=True,
                    offset=0.25,
                    move=0.2,
                    box_showfliers=False,
                )
            else:
                raise ValueError(f"Unknown motion name: {target_robot}")

            ax.set_ylabel("")

            ax.set_xlim(scale_mapping[target_robot][0], scale_mapping[target_robot][1])
            ax.set_title(f"Spatial Relationship Preservation of {robot_name_mapping[target_robot]}\nAcross Methods and Motions")

            # Apply font tuning for this plot
            _tune_raincloud_fonts(
                ax,
                title_size=title_size,
                label_size=label_size,
                tick_label_size=tick_label_size,
                legend_size=legend_size,
            )
            f.set_tight_layout(True)
            f.savefig(f"results/{target_robot}_gmr_phc_vs_mahr_spearman_across_motions.png", dpi=300)
            plt.show()
    # ======================================================================
    #
    # # ======================================================================
    # # ==== PHC vs MAHR Spearman's Coefficient Difference Across Motions =====
    # # choose UnitreeG1 robot for analysis
    # print("Plotting PHC vs MAHR Spearman's Coefficient Difference Across Motions On Unitree G1...")
    # target_robot = "UnitreeG1"
    # query = f"""
    # SELECT motion_name, phc_vs_mahr_diff
    # FROM spearman_comparison
    # WHERE phc_vs_mahr_diff IS NOT NULL AND robot_name = '{target_robot}'
    # """
    # df = pd.read_sql_query(query, analysis_conn)
    # motion_names = df['motion_name'].unique()
    # # average over motion_names for phc_vs_mahr_diff
    # motion_diff_dists = {}
    #
    # for motion_name in motion_names:
    #     df_motion = df[df['motion_name'] == motion_name]
    #     motion_diff_dfs = df_motion['phc_vs_mahr_diff']
    #     short_motion_name = shorten_motion_name(motion_name)
    #     motion_diff_dists[short_motion_name] = motion_diff_dfs.tolist()
    #
    # # convert to DataFrame
    # df = pd.DataFrame([(k, v) for k, values in motion_diff_dists.items() for v in values], columns=["motion_name", "Spearman_Difference"])
    # plt.figure(figsize=(10, 5))
    # flierprops = dict(marker='^', markerfacecolor='red', markersize=3, linestyle='none')
    # sns.boxplot(x="motion_name", hue="motion_name", y="Spearman_Difference", data=df, palette="Set3", flierprops=flierprops)
    # plt.axhline(0, color='red', linestyle='--')
    #
    # # try to tilt the x labels for better visibility
    # plt.xticks(rotation=45, ha='center')
    # # the text is out of the border of the figure, so adjust the bottom margin
    # plt.subplots_adjust(bottom=0.25)
    # plt.title("PHC vs MAHR Spearman's Coefficient Difference Across Motions On Unitree G1")
    # plt.ylabel("Difference of Spearman's Coefficient")
    # plt.xlabel(None)
    # plt.grid(True)
    # plt.show()
    # # ======================================================================

    source_conn.close()
    analysis_conn.close()


def generate_comparison_visualization(robot, frame_idx, motion_name, mahr_motion_file, gmr_motion_file, phc_motion_file, mahr_value, gmr_value, phc_value, gmr_vs_mahr_diff, phc_vs_mahr_diff):
    urdf_path = ROBOT_URDF_PATHS.get(robot)
    if not urdf_path or not os.path.isfile(urdf_path):
        print(f"[viz] Missing URDF for robot '{robot}'.")
        return None, None, None

    try:
        import mujoco  # type: ignore
    except ImportError as exc:
        print(f"[viz] MuJoCo is required for visualization but is not available: {exc}")
        return None, None, None

    try:
        from visualize_retargeted_motion import (
            apply_frame_to_state,
            build_scene_model,
            find_free_joint,
            _configure_tracking_mjv_camera,
            default_camera_offset_based_on_initial_frame,
            joint_qpos_addresses,
            joint_sizes,
            load_trajectory,
        )
    except Exception as exc:  # pragma: no cover - defensive import guard
        print(f"[viz] Unable to import visualization helpers: {exc}")
        return None, None, None

    try:
        frame_idx = int(frame_idx)
    except (TypeError, ValueError):
        frame_idx = 0
    frame_idx = max(frame_idx, 0)

    render_width, render_height = 640, 480
    try:
        model = build_scene_model(Path(urdf_path), render_width, render_height)
    except Exception as exc:
        print(f"[viz] Failed to build MuJoCo model for {robot}: {exc}")
        return None, None, None

    free_joint = find_free_joint(model)
    qpos_lookup = joint_qpos_addresses(model)
    size_lookup = joint_sizes(model)

    def _render_single_frame(motion_path_str: str | None, camera_offset: np.ndarray | None = None):
        if not motion_path_str:
            return None
        motion_path = Path(os.path.expanduser(os.path.expandvars(motion_path_str)))
        if not motion_path.is_file():
            print(f"[viz] Motion file not found: {motion_path}")
            return None
        try:
            _, frames = load_trajectory(motion_path)
        except Exception as exc:  # pragma: no cover - runtime guard
            print(f"[viz] Failed to load {motion_path.name}: {exc}")
            return None
        if not frames:
            print(f"[viz] Empty trajectory in {motion_path}")
            return None
        idx = min(frame_idx, len(frames) - 1)
        first_root_orient = frames[0].get("RootOrient") if isinstance(frames[0], dict) else None
        if camera_offset is None:
            if first_root_orient is not None:
                try:
                    camera_offset = default_camera_offset_based_on_initial_frame(first_root_orient)
                except Exception as exc:
                    print(f"[viz] Failed to derive camera offset from RootOrient: {exc}")
            else:
                camera_offset = np.array([2.5, 1.0, 0.5], dtype=float)
        data = mujoco.MjData(model)
        base_qpos = np.array(data.qpos, copy=True)
        apply_frame_to_state(frames[idx], base_qpos, data, qpos_lookup, size_lookup, free_joint)
        camera = mujoco.MjvCamera()
        cam_updater = _configure_tracking_mjv_camera(camera, data, free_joint, camera_offset)
        if cam_updater:
            cam_updater()
        renderer = mujoco.Renderer(model, render_height, render_width)
        renderer.update_scene(data, camera=camera)
        frame_rgb = renderer.render().copy()
        del renderer
        return frame_rgb, camera_offset

    camera_offset = None
    mahr_frame_image, camera_offset = _render_single_frame(mahr_motion_file, camera_offset)
    gmr_frame_image, camera_offset = _render_single_frame(gmr_motion_file, camera_offset)
    phc_frame_image, camera_offset = _render_single_frame(phc_motion_file, camera_offset)

    frames_to_plot = [
        (name, img)
        for name, img in ((f"MAHR\n{mahr_value:.4f}", mahr_frame_image), (f"GMR\n{gmr_value:.4f}({gmr_vs_mahr_diff:+.4f})", gmr_frame_image), (f"PHC\n{phc_value:.4f}({phc_vs_mahr_diff:+.4f})", phc_frame_image))
        if img is not None
    ]
    if frames_to_plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            print("[viz] matplotlib is not installed; skipping comparison figure.")
        else:
            cols = 2
            rows = 2
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

            axes[0, 0].set_title("Original Motion")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(frames_to_plot[0][1])
            axes[0, 1].set_title(frames_to_plot[0][0])
            axes[0, 1].axis("off")

            axes[1, 0].imshow(frames_to_plot[1][1])
            axes[1, 0].set_title(frames_to_plot[1][0])
            axes[1, 0].axis("off")

            axes[1, 1].imshow(frames_to_plot[2][1])
            axes[1, 1].set_title(frames_to_plot[2][0])
            axes[1, 1].axis("off")
            # for ax, (label, img) in zip(axes, frames_to_plot):
            #     ax.imshow(img)
            #     ax.set_title(label)
            #     ax.axis("off")
            fig.suptitle(f"{robot} – Motion {motion_name} - Frame {frame_idx}")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
            plt.close(fig)
    else:
        print(f"[viz] Unable to visualize frame {frame_idx} for {robot} (no valid motion files).")

    return mahr_frame_image, gmr_frame_image, phc_frame_image


def compare_frames(db_path, analysis_db_path):
    source_conn = open_db(db_path)
    analysis_conn = open_db(analysis_db_path)

    # query to get all unique motion_name, robot_name, frame_idx combinations where gmr_vs_mahr_diff and phc_vs_mahr_diff are both not null
    query = """
    SELECT DISTINCT motion_name, robot_name, frame_idx, gmr_vs_mahr_diff, phc_vs_mahr_diff, mahr_motion_file, gmr_motion_file, phc_motion_file, mahr, gmr, phc
    FROM spearman_comparison
    WHERE gmr_vs_mahr_diff IS NOT NULL AND phc_vs_mahr_diff IS NOT NULL
    """
    df_unique = pd.read_sql_query(query, analysis_conn)
    # for every unique motion_name and robot_name, find the top 3 frames with largest gmr_vs_mahr_diff and phc_vs_mahr_diff by sorting the dataframe
    top_n = 1
    for (motion_name, robot_name), group in df_unique.groupby(['motion_name', 'robot_name']):
        group_highest_gmr = group.sort_values(by='gmr_vs_mahr_diff', ascending=False).head(top_n)
        group_highest_phc = group.sort_values(by='phc_vs_mahr_diff', ascending=False).head(top_n)
        group_lowest_gmr = group.sort_values(by='gmr_vs_mahr_diff', ascending=True).head(top_n)
        group_lowest_phc = group.sort_values(by='phc_vs_mahr_diff', ascending=True).head(top_n)
        print("============================================================")
        print(f"Motion: {motion_name}, Robot: {robot_name}")
        print(f"Top {top_n} frames with HIGHEST GMR vs MAHR difference:")
        for _, row in group_highest_gmr.iterrows():
            robot = row['robot_name']
            frame_idx = row['frame_idx']
            motion_name = row["motion_name"]
            mahr_motion_file = row['mahr_motion_file']
            gmr_motion_file = row['gmr_motion_file']
            phc_motion_file = row['phc_motion_file']
            mahr_value = row['mahr']
            gmr_value = row['gmr']
            phc_value = row['phc']
            gmr_vs_mahr_diff = row['gmr_vs_mahr_diff']
            phc_vs_mahr_diff = row['phc_vs_mahr_diff']
            if abs(gmr_vs_mahr_diff) > 0.1 or abs(phc_vs_mahr_diff) > 0.1:
                generate_comparison_visualization(robot, frame_idx, motion_name, mahr_motion_file, gmr_motion_file, phc_motion_file, mahr_value, gmr_value, phc_value, gmr_vs_mahr_diff, phc_vs_mahr_diff)

        print(f"Top {top_n} frames with HIGHEST PHC vs MAHR difference:")
        for _, row in group_highest_phc.iterrows():
            robot = row['robot_name']
            frame_idx = row['frame_idx']
            motion_name = row["motion_name"]
            mahr_motion_file = row['mahr_motion_file']
            gmr_motion_file = row['gmr_motion_file']
            phc_motion_file = row['phc_motion_file']
            mahr_value = row['mahr']
            gmr_value = row['gmr']
            phc_value = row['phc']
            gmr_vs_mahr_diff = row['gmr_vs_mahr_diff']
            phc_vs_mahr_diff = row['phc_vs_mahr_diff']
            if abs(gmr_vs_mahr_diff) > 0.1 or abs(phc_vs_mahr_diff) > 0.1:
                generate_comparison_visualization(robot, frame_idx, motion_name, mahr_motion_file, gmr_motion_file, phc_motion_file, mahr_value, gmr_value, phc_value, gmr_vs_mahr_diff, phc_vs_mahr_diff)

        print(f"Top {top_n} frames with LOWEST GMR vs MAHR difference:")
        for _, row in group_lowest_gmr.iterrows():
            robot = row['robot_name']
            frame_idx = row['frame_idx']
            motion_name = row["motion_name"]
            mahr_motion_file = row['mahr_motion_file']
            gmr_motion_file = row['gmr_motion_file']
            phc_motion_file = row['phc_motion_file']
            mahr_value = row['mahr']
            gmr_value = row['gmr']
            phc_value = row['phc']
            gmr_vs_mahr_diff = row['gmr_vs_mahr_diff']
            phc_vs_mahr_diff = row['phc_vs_mahr_diff']
            if abs(gmr_vs_mahr_diff) > 0.1 or abs(phc_vs_mahr_diff) > 0.1:
                generate_comparison_visualization(robot, frame_idx, motion_name, mahr_motion_file, gmr_motion_file, phc_motion_file, mahr_value, gmr_value, phc_value, gmr_vs_mahr_diff, phc_vs_mahr_diff)
        print(f"Top {top_n} frames with LOWEST PHC vs MAHR difference:")
        for _, row in group_lowest_phc.iterrows():
            robot = row['robot_name']
            frame_idx = row['frame_idx']
            motion_name = row["motion_name"]
            mahr_motion_file = row['mahr_motion_file']
            gmr_motion_file = row['gmr_motion_file']
            phc_motion_file = row['phc_motion_file']
            mahr_value = row['mahr']
            gmr_value = row['gmr']
            phc_value = row['phc']
            gmr_vs_mahr_diff = row['gmr_vs_mahr_diff']
            phc_vs_mahr_diff = row['phc_vs_mahr_diff']
            if abs(gmr_vs_mahr_diff) > 0.1 or abs(phc_vs_mahr_diff) > 0.1:
                generate_comparison_visualization(robot, frame_idx, motion_name, mahr_motion_file, gmr_motion_file, phc_motion_file, mahr_value, gmr_value, phc_value, gmr_vs_mahr_diff, phc_vs_mahr_diff)



if __name__ == "__main__":
    # parse commandline arguments using argparser
    parser = argparse.ArgumentParser(description="Process motion data")
    parser.add_argument("--motion_name", type=str, default="144_32_poses")
    parser.add_argument("--robot_name", type=str, default="UnitreeG1")
    parser.add_argument("--method_name", type=str, default="mahr")
    parser.add_argument("--db_path", type=str, default=get_absolute_path("log/metrics.sqlite"), help="Path to SQLite DB to store results")
    parser.add_argument("--analysis_db_path", type=str, default=get_absolute_path("log/analysis.sqlite"), help="Path to SQLite DB to store results")
    parser.add_argument("--write_db", action="store_true", help="Write summary stats to the database, one row per frame")
    parser.add_argument("--mode", type=str, default="data-gen")

    args = parser.parse_args()

    motion_name = args.motion_name
    robot_name = args.robot_name
    method_name = args.method_name

    if args.mode == "data-gen":
        data_generation(motion_name, robot_name, method_name, db_path=args.db_path, write_db=args.write_db)
    elif args.mode == "analysis":
        analysis(db_path=args.db_path, analysis_db_path=args.analysis_db_path)
    elif args.mode == "plot":
        plot(db_path=args.db_path, analysis_db_path=args.analysis_db_path)
    elif args.mode == "frame-compare":
        compare_frames(db_path=args.db_path, analysis_db_path=args.analysis_db_path)
    else:
        raise ValueError(f"Unknown mode {args.mode}")
