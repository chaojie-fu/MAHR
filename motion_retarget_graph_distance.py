"""
motion_retarget_graph_distance.py

This script provides methods for retargeting motion capture data to robot configurations using graph distance optimization.
It supports multiprocessing, caching, and sensitivity analysis, and can process both CMU and AMASS datasets.

Main features:
- Retargets mocap data to various robot models using graph distance optimization.
- Supports multiprocessing for efficient batch processing.
- Handles caching of motion and graph data for speed.
- Provides utility functions for edge filtering, logging, and configuration.
- Includes main entrypoints for different retargeting scenarios.
"""

import datetime
import inspect
import json
import logging
import multiprocessing
import os
import pathlib
import pickle
import time
from copy import deepcopy

import matplotlib
import multiprocessing_logging
import numpy as np
import scipy.optimize
import torch

from config.RobotConfig.AgibotX1Config import Agibot_X1_WholeBody_Config
from config.RobotConfig.FourierGR1Config import Fourier_GR1_WholeBody_Config
from config.RobotConfig.BerkeleyHumanoidConfig import Berkeley_Humanoid_Lite_WholeBody_Config
from config.RobotConfig.BostonDynamicsAtlasHDConfig import Boston_Dynamics_Atlas_HD_WholeBody_Config
from config.RobotConfig.FourierN1Config import Fourier_N1_WholeBody_Config
from config.RobotConfig.UnitreeG1 import Unitree_G1_WholeBody_Config
from config.RobotConfig.UnitreeH1 import Unitree_H1_WholeBody_Config
from config.MocapConfig.CMUAsfAmcConfig import CMU_AsfAmc_WholeBody_Config
from config.MocapConfig.AmassSmplHGConfig import AMASS_DanceDB_SMPL_HG_WholeBody_Config
from config.RobotConfig.iCubConfig import iCub_WholeBody_Config
from mocap_data_converter.config.MocapConfig import MocapConfig
from mocap_data_converter.config.RobotConfig import RobotConfig
from mocap_data_converter.utils.graph_distance.compute_graph_distance import compute_vector_graph_distance
from mocap_data_converter.utils.graph_distance.file_index import generate_file_index
from mocap_data_converter.utils.graph_distance.mocap_graph_generator import generate_vector_graphs_for_motion
from mocap_data_converter.utils.graph_distance.robot_graph_generator import generate_vector_symbolic_graph_for_robot
from mocap_data_converter.utils.poselib.core.rotation3d import rot_matrix_from_quaternion

NUM_WORKERS = 120

# =========================
# Utility and Helper Functions
# =========================


def get_asf_amc_mocap(file_index, subject_idx, motion_idx):
    """
    Retrieve ASF/AMC file paths and metadata for a given subject and motion index.

    :param file_index: List of file index dictionaries.
    :param subject_idx: Subject identifier.
    :param motion_idx: Motion identifier.
    :return: Tuple of (asf_file, amc_file, fps, description, tag).
    """
    file_name = f"{str(subject_idx).zfill(2)}_{str(motion_idx).zfill(2)}"

    asf_file = None
    amc_file = None
    fps = None
    description = None
    tag = None
    found_item = False
    for item in file_index:
        if f"{file_name}" in item["amc"]:
            asf_file = item["asf"]
            amc_file = item["amc"]
            fps = item["fps"]
            description = item["description"]
            tag = item["tag"]
            found_item = True
            break

    if not found_item:
        raise ValueError(f"Source mocap data not found for motion {file_name}")
    else:
        return asf_file, amc_file, fps, description, tag


def filter_kwargs_by_signature(func, args_dict):
    """
    Filter a dictionary to keep only valid parameters according to a function's signature.

    :param func: Target function/method.
    :param args_dict: Original parameter dictionary.
    :return: Filtered dictionary with valid parameters.
    """
    # Get function signature
    sig = inspect.signature(func)

    # Collect valid parameter names
    valid_params = []
    has_kwargs = False

    for param in sig.parameters.values():
        if param.kind in (
            param.POSITIONAL_OR_KEYWORD,
            param.KEYWORD_ONLY
        ):
            valid_params.append(param.name)
        elif param.kind == param.VAR_KEYWORD:
            has_kwargs = True  # Flag if function accepts **kwargs

    # Return original dict if **kwargs exists
    if has_kwargs:
        return args_dict.copy()

    # Filter invalid parameters
    return {k: v for k, v in args_dict.items() if k in valid_params}


def _setup_logger():
    """
    Set up the logger for both file and stream output, and install multiprocessing handler.
    """
    # ====================================================================================================
    # Logger
    startTime = datetime.datetime.now().strftime("%Y_%b_%d-%H_%M_%S")

    # Create and configure logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s]: %(message)s')

    os.makedirs("./log", exist_ok=True)
    file_handler = logging.FileHandler(filename=f"./log/motion_retarget_graph_distance_{startTime}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    multiprocessing_logging.install_mp_handler()
    # ====================================================================================================


def _cached_motion_config(_cache_dir, _mocap_config, _bvh_motion_file, _norm_pose_dir, _motion_range=None):
    """
    Load or cache the mocap configuration for a given BVH motion file.

    :param _cache_dir: Directory to store cache files.
    :param _mocap_config: MocapConfig instance.
    :param _bvh_motion_file: Path to the BVH motion file.
    :param _norm_pose_dir: Directory for normalized pose.
    :param _motion_range: Optional motion range.
    :return: Loaded or cached MocapConfig instance.
    """
    cache_file = pathlib.Path(os.path.join(
        _cache_dir,
        "mocap_config",
        f"mc_cache_{pathlib.Path(_bvh_motion_file).stem}_{_motion_range}.pkl"))
    os.makedirs(os.path.join(_cache_dir, "mocap_config"), exist_ok=True)

    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        _mocap_config.load_motion(_bvh_motion_file, _norm_pose_dir, motion_range=_motion_range)

        with open(cache_file, 'wb') as f:
            pickle.dump(_mocap_config, f)
        return _mocap_config

# =========================
# Optimization Functions
# =========================


def _construct_objective_function(
    mocap_graph_at_frame_idx,
    mocap_config,
    robot_graph_fn,
    robot_config,
    root_quat,
    flag_weighted_optimization_by_closeness,
    flag_filter_edges_when_generating_graph,
    edge_filter
):
    """
    Construct the objective function for optimization using a callable robot graph generator.

    :return: Objective function mapping decision variables to graph distance.
    """
    root_mat = rot_matrix_from_quaternion(torch.from_numpy(root_quat)).numpy()

    def objective(x):
        gc = x[:-1]
        root_height = x[-1]
        robot_graph = robot_graph_fn(gc, root_mat, root_height)
        return compute_vector_graph_distance(
            mocap_graph_at_frame_idx,
            mocap_config.key_bodies,
            robot_graph,
            robot_config.key_joints,
            flag_weighted_optimization_by_closeness=flag_weighted_optimization_by_closeness,
            flag_filter_edges_when_generating_graph=flag_filter_edges_when_generating_graph,
            edge_filter=edge_filter
        )

    return objective


def _construct_default_optim_kwargs(
    ground_in_graph,
    edge_filters,
    force_generate_motion_dict=True,
    force_generate_graph=True,
    filter_edges_when_generating_graph=True,
    normalize_edge_weights_using_T_pose=True,
    use_mp=False,
    mp_level="trajectory",
    weighted_optimization_by_closeness=True
):
    """
    Construct default optimization keyword arguments.

    :param ground_in_graph: Whether ground height is considered in optimization.
    :param edge_filters: Filtered graph edges for optimization.
    :param force_generate_motion_dict: regenerate the motion_dict from asf/amc files
    :param force_generate_graph: regenerate the graph from motion dict file
    :param filter_edges_when_generating_graph: filter edges when generating graphs
    :param normalize_edge_weights_using_T_pose: normalize the edges using T-pose
    :param use_mp: use multiprocessing for graph generation
    :param mp_level: Use multiprocessing at which level, can be either "trajectory" or "frame"
    :param weighted_optimization_by_closeness: Optimize graph using weighted edge by closeness
    :return:
    """
    return {
        "flag_force_generate_motion_dict": force_generate_motion_dict,
        "flag_force_generate_graph": force_generate_graph,
        "flag_filter_edges_when_generating_graph": filter_edges_when_generating_graph,
        "flag_normalize_edge_weights_using_T_pose": normalize_edge_weights_using_T_pose,
        "flag_use_mp": use_mp,
        "mp_level": mp_level,
        "flag_weighted_optimization_by_closeness": weighted_optimization_by_closeness,
        "flag_ground_in_graph": ground_in_graph,
        "edge_filters": edge_filters
    }


def _calculate_variable_bounds(gc_bounds, x0, constraint_x_by_x0):
    """
    Calculate variable bounds for optimization.

    :return: scipy.optimize.Bounds object.
    """
    if constraint_x_by_x0:
        variable_lower_bounds = np.maximum(
            x0 - 10.0 / 120.0,
            np.concatenate([gc_bounds[0], np.asarray([0.0])], axis=0)
        )
        variable_higher_bounds = np.minimum(
            x0 + 10.0 / 120.0,
            np.concatenate([gc_bounds[1], np.asarray([4.0])], axis=0),
        )
    else:
        variable_lower_bounds = np.concatenate([gc_bounds[0], np.asarray([0.0])], axis=0)
        variable_higher_bounds = np.concatenate([gc_bounds[1], np.asarray([4.0])], axis=0)
    return scipy.optimize.Bounds(variable_lower_bounds, variable_higher_bounds)


def _log_optimization_result(frame_idx, mocap_config, obj_construct_time, v0, eval_time, optim_time, res, optimize_basis, total_time):
    """
    Log the optimization result.
    """
    logging.getLogger().info(
        f"\n"
        f"Optimization result for frame {frame_idx} of motion {mocap_config.name}\n"
        f"Objective construction time: {obj_construct_time:.4f}\n"
        f"Initial objective value: {v0}\n"
        f"Evaluation time of objective: {eval_time:.4f}\n"
        f"Optimized objective value: {optimize_basis(res.x)}\n"
        f"Optimized joint angles: {res.x}\n"
        f"Optimization time: {optim_time:.4f}\n"
        f"Total time: {total_time:.4f}"
    )


def optimize_posture_for_vector_graph(
    frame_idx,
    mocap_config: MocapConfig,
    robot_config: RobotConfig,
    mocap_graph_at_frame_idx,
    robot_graph_fn,
    mocap_t_pose_graph,
    robot_t_pose_graph,
    root_quat,
    gc_bounds,
    x0,
    trajectory,
    cache_dir,
    flag_weighted_optimization_by_closeness=False,
    flag_filter_edges_when_generating_graph=False,
    edge_filter=None,
    constraint_x_by_x0=True
):
    """
    Optimize robot joint angles for a single frame to minimize the graph distance.

    :param frame_idx: Index of the frame to optimize.
    :param mocap_config: MocapConfig instance.
    :param robot_config: RobotConfig instance.
    :param mocap_graph_at_frame_idx: Mocap graph at the current frame.
    :param robot_graph_fn: Callable returning the robot graph for provided joint angles.
    :param mocap_t_pose_graph: Mocap T-pose graph.
    :param robot_t_pose_graph: Robot T-pose graph.
    :param root_quat: Root quaternion.
    :param gc_bounds: Bounds for generalized coordinates.
    :param x0: Initial guess for optimization.
    :param trajectory: Dictionary to store optimized trajectory.
    :param cache_dir: Directory for caching results.
    :param flag_weighted_optimization_by_closeness: Whether to use weighted optimization.
    :param flag_filter_edges_when_generating_graph: Whether to filter edges.
    :param edge_filter: Edge filter specification.
    :param constraint_x_by_x0: Whether to constrain optimization by x0.
    :return: Optimized joint angles as a numpy array.
    """
    logger = logging.getLogger()
    cache_file = os.path.join(cache_dir, f"{frame_idx}.pkl")
    cache_value_file = os.path.join(cache_dir, f"{frame_idx}_value.pkl")

    logger.info(f"Start optimizing robot joint angles for frame {frame_idx}")

    if os.path.isfile(cache_file):
        logger.info(f"Found existing optimized value for frame {frame_idx}, loading")
        with open(cache_file, 'rb') as file:
            x0 = pickle.load(file)
    else:
        logger.info(f"Optimizing frame No. {frame_idx} of motion {mocap_config.name}")
        start_time = time.time()
        obj_construct_time_start = time.time()

        # Construct the objective function
        f = _construct_objective_function(
            mocap_graph_at_frame_idx,
            mocap_config,
            robot_graph_fn,
            robot_config,
            root_quat,
            flag_weighted_optimization_by_closeness,
            flag_filter_edges_when_generating_graph,
            edge_filter
        )

        obj_construct_time_end = time.time()

        # Calculate bounds
        x_bounds = _calculate_variable_bounds(gc_bounds, x0, constraint_x_by_x0)

        # Define the optimization basis function
        def optimize_basis(x):
            return f(x) + 1e-3 * np.sum(np.square(np.clip(x, -0.78, 0.78)))

        # Run optimization
        optim_start_time = time.time()
        if constraint_x_by_x0:
            logger.info("Local Optimizing")
            res = scipy.optimize.differential_evolution(optimize_basis, x_bounds, maxiter=200)
        else:
            logger.info("Global Optimizing")
            res = scipy.optimize.differential_evolution(
                optimize_basis, x_bounds,
                init="sobol",
                strategy="best1bin",
                maxiter=5000,
                tol=1e-4,
            )
        optim_end_time = time.time()
        eval_start_time = time.time()
        v0 = optimize_basis(x0)
        eval_end_time = time.time()

        # Log optimization result
        _log_optimization_result(
            frame_idx, mocap_config,
            obj_construct_time_end - obj_construct_time_start,
            v0,
            eval_end_time - eval_start_time,
            optim_end_time - optim_start_time,
            res,
            optimize_basis,
            time.time() - start_time
        )

        x0 = res.x

        # Cache results
        with open(cache_file, 'wb') as file:
            pickle.dump(x0, file)
        with open(cache_value_file, 'wb') as file:
            pickle.dump(optimize_basis(x0), file)

    # Store results in trajectory
    for joint_idx, joint_name in enumerate(robot_config.robot.actuated_joint_names):
        trajectory[joint_name][frame_idx] = float(x0[joint_idx])
    trajectory["root_height"][frame_idx] = float(x0[-1])

    return x0


# =========================
# Main Retargeting and Analysis Functions
# =========================


def motion_retarget_graph_distance(
    cache_dir,
    mocap_config: MocapConfig,
    robot_config: RobotConfig,
    flag_force_generate_motion_dict=False,
    flag_force_generate_graph=False,
    flag_filter_edges_when_generating_graph=False,
    edge_filters=None,
    flag_normalize_edge_weights_using_T_pose=True,
    flag_use_mp=False,
    mp_level="trajectory",
    flag_weighted_optimization_by_closeness=False
):
    """
    Main function to retarget motion using graph distance optimization.

    :param cache_dir: Directory for caching and output.
    :param mocap_config: MocapConfig instance.
    :param robot_config: RobotConfig instance.
    :param flag_force_generate_motion_dict: Force regeneration of motion dict.
    :param flag_force_generate_graph: Force regeneration of graph.
    :param flag_filter_edges_when_generating_graph: Whether to filter edges.
    :param edge_filters: Edge filter specification.
    :param flag_normalize_edge_weights_using_T_pose: Normalize edge weights using T-pose.
    :param flag_use_mp: Whether to use multiprocessing.
    :param mp_level: Multiprocessing level ("trajectory" or "frame").
    :param flag_weighted_optimization_by_closeness: Use weighted optimization.
    :return: None
    """
    logging.getLogger().info(f"Start retargeting motion {mocap_config.name}")
    # ====================================================================================================
    # define directories
    os.makedirs(cache_dir, exist_ok=True)

    motion_dict_dir = os.path.join(cache_dir, "motion_dict")
    os.makedirs(motion_dict_dir, exist_ok=True)

    logging.getLogger().info(f"temporary files for motion dictionary will be saved to {motion_dict_dir}")

    if flag_normalize_edge_weights_using_T_pose:
        graph_dir = os.path.join(cache_dir, "normalized_vector_graph_output")
    else:
        graph_dir = os.path.join(cache_dir, "vector_graph_output")

    os.makedirs(graph_dir, exist_ok=True)
    logging.getLogger().info(f"temporary graphs will be saved to {graph_dir}")
    # ====================================================================================================

    # ====================================================================================================
    # construct normalized graph for robot and optimize it
    (_, gc_bounds, robot_t_pose_graph, robot_graph_fn,
     robot_t_pose_toes_horizontal_distance) = (
        generate_vector_symbolic_graph_for_robot(
            robot_config,
            normalize_edge_weights=flag_normalize_edge_weights_using_T_pose,
            filter_edges_when_generating_graph=flag_filter_edges_when_generating_graph,
            edge_filter=edge_filters))

    # ====================================================================================================

    # ====================================================================================================
    # construct graph from asf/amc files and normalize the graph weight using T-pose
    file_name = mocap_config.name

    # asf_file, amc_file, fps, description, tag = get_asf_amc_mocap(file_index, subject_idx, motion_idx)

    t_pose_graph, mocap_graphs, motion_global_dict, mocap_t_pose_toes_horizontal_distance = (
        generate_vector_graphs_for_motion(
            mocap_config,
            motion_dict_dir,
            graph_dir,
            normalize_edge_weights=flag_normalize_edge_weights_using_T_pose,
            force_generate_motion_dict=flag_force_generate_motion_dict,
            force_generate_graph=flag_force_generate_graph,
            filter_edges_when_generating_graph=flag_filter_edges_when_generating_graph,
            edge_filter=edge_filters))
    # ====================================================================================================

    assert len(mocap_config.key_bodies) == len(robot_config.key_joints), \
        "Mocap and Robot don't have same number of key bodies"

    variable_lower_bounds = np.concatenate([gc_bounds[0], np.asarray([0.0])], axis=0)
    variable_higher_bounds = np.concatenate([gc_bounds[1], np.asarray([4.0])], axis=0)
    x0 = variable_lower_bounds + (variable_higher_bounds - variable_lower_bounds) * 0.5

    optimized_dir = os.path.join(cache_dir, "optimized_trajectories_vector_graph", f"{file_name}")

    os.makedirs(optimized_dir, exist_ok=True)
    logging.getLogger().info(f"optimized joint angles will be saved to {optimized_dir}")

    # num_frames_to_compute = 100
    num_frames_to_compute = len(mocap_graphs)
    logging.getLogger().info(f"Start retargeting, totally {num_frames_to_compute} frames to be optimized")

    # remove redundant class members from class to accelerate pickle
    simplified_mocap_config = deepcopy(mocap_config)
    simplified_mocap_config.uni_repr = None
    simplified_mocap_config.norm_pose_repr = None

    if flag_use_mp and mp_level == "frame":
        with multiprocessing.Manager() as manager:
            shared_trajectory = manager.dict()
            for joint in robot_config.robot.actuated_joint_names:
                # elements of dict should be proxy objects as well,
                # otherwise the value of the elements will not be modified in child processes
                joint_list = manager.list()
                for idx in range(num_frames_to_compute):
                    joint_list.append(0.0)
                shared_trajectory[joint] = joint_list
            root_height_list = manager.list()
            for idx in range(num_frames_to_compute):
                root_height_list.append(0.0)
            shared_trajectory["root_height"] = root_height_list

            with multiprocessing.pool.Pool(processes=NUM_WORKERS, maxtasksperchild=1) as pool:
                for frame_idx in range(num_frames_to_compute):
                    pool.apply_async(
                        optimize_posture_for_vector_graph,
                        (frame_idx,
                         simplified_mocap_config,
                         robot_config,
                         mocap_graphs[frame_idx],
                        robot_graph_fn,
                        t_pose_graph,
                        robot_t_pose_graph,
                        motion_global_dict["global_rotations"][frame_idx][0],
                        gc_bounds,
                        x0,
                         shared_trajectory,
                         optimized_dir,),
                        {"flag_weighted_optimization_by_closeness": flag_weighted_optimization_by_closeness,
                         "flag_filter_edges_when_generating_graph": flag_filter_edges_when_generating_graph,
                         "edge_filter": edge_filters,
                         "constraint_x_by_x0": False}
                    )
                pool.close()
                pool.join()

            # copy data from the shared object to normal python object
            trajectory = {joint: [0] * num_frames_to_compute for joint in robot_config.robot.actuated_joint_names}
            trajectory["root_height"] = [0] * num_frames_to_compute
            for joint in robot_config.robot.actuated_joint_names:
                for frame in range(num_frames_to_compute):
                    trajectory[joint][frame] = shared_trajectory[joint][frame]
            for frame in range(num_frames_to_compute):
                trajectory["root_height"][frame] = shared_trajectory["root_height"][frame]
    else:
        trajectory = {joint: [0] * num_frames_to_compute for joint in robot_config.robot.actuated_joint_names}
        trajectory["root_height"] = [0] * num_frames_to_compute

        x0_initialized = False
        for frame_idx in range(num_frames_to_compute):
            x0 = optimize_posture_for_vector_graph(
                frame_idx,
                simplified_mocap_config,
                robot_config,
                mocap_graphs[frame_idx],
                robot_graph_fn,
                t_pose_graph,
                robot_t_pose_graph,
                motion_global_dict["global_rotations"][frame_idx][0],
                gc_bounds,
                x0,
                trajectory,
                optimized_dir,
                flag_weighted_optimization_by_closeness=flag_weighted_optimization_by_closeness,
                flag_filter_edges_when_generating_graph=flag_filter_edges_when_generating_graph,
                edge_filter=edge_filters,
                constraint_x_by_x0=x0_initialized
            )
            # x0_initialized = True

    # save optimized data to json file
    motion_file = os.path.join(optimized_dir, f"{file_name}_gd.json")
    motion_file_v2 = os.path.join(optimized_dir, f"{file_name}_gd_v2.json")
    motion_json = {
        "fps": mocap_config.uni_repr.fps,
        "trajectory": []
    }
    motion_json_v2 = {
        "fps": mocap_config.uni_repr.fps,
        "trajectory": []
    }

    horizontal_ratio = robot_t_pose_toes_horizontal_distance / mocap_t_pose_toes_horizontal_distance
    for frame in range(num_frames_to_compute):
        frame_dict = {
            "RootPos": [0.0] * 3,
            "RootOrient": [0.0] * 4,
            "JointPos": [0.0] * len(robot_config.robot.actuated_joint_names)
        }
        frame_dict_v2 = {
            "RootPos": [0.0] * 3,
            "RootOrient": [0.0] * 4,
            "JointPos": {name: 0.0 for name in robot_config.robot.actuated_joint_names}
        }
        frame_dict["RootPos"][0] = float(motion_global_dict["global_translations"][frame][0][0]) * horizontal_ratio
        frame_dict["RootPos"][1] = float(motion_global_dict["global_translations"][frame][0][1]) * horizontal_ratio
        frame_dict["RootPos"][2] = trajectory["root_height"][frame]

        frame_dict_v2["RootPos"][0] = float(motion_global_dict["global_translations"][frame][0][0]) * horizontal_ratio
        frame_dict_v2["RootPos"][1] = float(motion_global_dict["global_translations"][frame][0][1]) * horizontal_ratio
        frame_dict_v2["RootPos"][2] = trajectory["root_height"][frame]

        for rot_idx in range(4):
            frame_dict["RootOrient"][rot_idx] = float(motion_global_dict["global_rotations"][frame][0][rot_idx])
            frame_dict_v2["RootOrient"][rot_idx] = float(motion_global_dict["global_rotations"][frame][0][rot_idx])

        for joint_idx, joint_name in enumerate(robot_config.robot.actuated_joint_names):
            output_idx = robot_config.index_mapping[joint_name]
            frame_dict["JointPos"][output_idx] = trajectory[joint_name][frame]
            frame_dict_v2["JointPos"][joint_name] = trajectory[joint_name][frame]
        motion_json["trajectory"].append(frame_dict)
        motion_json_v2["trajectory"].append(frame_dict_v2)
    with open(motion_file, "w") as f:
        json.dump(motion_json, f, indent=2)

    with open(motion_file_v2, "w") as f:
        json.dump(motion_json_v2, f, indent=2)


def execute_retargeting_asf_amc(root_dir, file_index, motion_trajs, mocap_config, robot_config, optim_kwargs):
    validated_optim_kwargs = filter_kwargs_by_signature(motion_retarget_graph_distance, optim_kwargs)
    if optim_kwargs["mp_level"] not in ["trajectory", "frame"]:
        raise ValueError(f"`mp_level` should be one of: 'trajectory', 'frame', "
                         f"but {optim_kwargs['flag_use_mp']} is provided")
    elif (optim_kwargs["mp_level"] == "trajectory") and optim_kwargs["flag_use_mp"]:
        validated_optim_kwargs["flag_use_mp"] = False
        with multiprocessing.Manager() as manager:
            with multiprocessing.pool.Pool(processes=NUM_WORKERS, maxtasksperchild=1) as pool:
                for [subject_idx, motion_idx] in motion_trajs:
                    pool.apply_async(motion_retarget_graph_distance,
                                     (
                                         root_dir,
                                         mocap_config.load_motion(
                                             *get_asf_amc_mocap(file_index, subject_idx, motion_idx)[0:3]),
                                         robot_config,
                                     ),
                                     validated_optim_kwargs)
                pool.close()
                pool.join()
    else:
        for [subject_idx, motion_idx] in motion_trajs:
            motion_retarget_graph_distance(root_dir,
                                           mocap_config.load_motion(
                                               *get_asf_amc_mocap(file_index, subject_idx, motion_idx)[0:3]
                                           ),
                                           robot_config,
                                           **validated_optim_kwargs)


def execute_retargeting_bvh(root_dir, bvh_root_dir, mocap_config, robot_config, optim_kwargs):
    validated_optim_kwargs = filter_kwargs_by_signature(motion_retarget_graph_distance, optim_kwargs)
    if optim_kwargs["mp_level"] not in ["trajectory", "frame"]:
        raise ValueError(f"`mp_level` should be one of: 'trajectory', 'frame', "
                         f"but {optim_kwargs['flag_use_mp']} is provided")
    elif (optim_kwargs["mp_level"] == "trajectory") and optim_kwargs["flag_use_mp"]:
        validated_optim_kwargs["flag_use_mp"] = False
        with multiprocessing.Manager() as manager:
            with multiprocessing.pool.Pool(processes=NUM_WORKERS, maxtasksperchild=1) as pool:
                for root, dirs, files in os.walk(bvh_root_dir):
                    for file in files:
                        if file.endswith(".bvh") and ("norm_pose" not in file):
                            pool.apply_async(motion_retarget_graph_distance,
                                             (
                                                 root_dir,
                                                 _cached_motion_config(
                                                     root_dir,
                                                     mocap_config,
                                                     os.path.join(root, file),
                                                     os.path.join(bvh_root_dir, "norm_pose")
                                                 ),
                                                 robot_config
                                             ),
                                             validated_optim_kwargs)
                pool.close()
                pool.join()
    else:
        for root, dirs, files in os.walk(bvh_root_dir):
            for file in files:
                if file.endswith(".bvh") and ("norm_pose" not in file):
                    motion_retarget_graph_distance(root_dir,
                                                   _cached_motion_config(
                                                       root_dir,
                                                       mocap_config,
                                                       os.path.join(root, file),
                                                       os.path.join(bvh_root_dir, "norm_pose")
                                                   ),
                                                   robot_config,
                                                   **validated_optim_kwargs)


def main_CMU_to_GR1():
    """
    Retarget CMU motion capture data to the GR1 robot using graph distance optimization.
    """
    _setup_logger()

    root_dir = "./graph_distance_CMU_to_GR1"
    os.makedirs(root_dir, exist_ok=True)

    # mocap options
    asf_amc_root_dir = "./mocap_data_converter/rsc/all_asfamc/subjects"
    index_file = "./mocap_data_converter/rsc/all_asfamc/index.json"

    # The pose used for normalization, should be one of "t_pose" and "stand"
    normalize_pose = "t_pose"

    # mocap config
    mocap_config = CMU_AsfAmc_WholeBody_Config(norm_pose_type=normalize_pose)

    # robot config
    robot_config = Fourier_GR1_WholeBody_Config(norm_pose_type=normalize_pose)

    assert (("ground" in mocap_config.key_bodies) and ("ground" in robot_config.key_joints)) or \
           (("ground" not in mocap_config.key_bodies) and ("ground" not in robot_config.key_joints))
    flag_ground_in_graph = (("ground" in mocap_config.key_bodies) and ("ground" in robot_config.key_joints))

    # edge filter that remove the edges between upper body and lower body
    edge_filters = []
    for start_idx in range(len(mocap_config.key_bodies)):
        for end_idx in range(start_idx + 1, len(mocap_config.key_bodies)):
            edge_filters.append([start_idx, end_idx])

    # generate file index
    file_index = generate_file_index(asf_amc_root_dir, index_file)
    with open(os.path.join(root_dir, "file_index.json"), "w") as f:
        json.dump(file_index, f, indent=2)

    motion_trajs = []
    for item in file_index:
        subject_idx, motion_idx = pathlib.Path(item["amc"]).stem.split("_")
        motion_trajs.append([int(subject_idx), int(motion_idx)])

    optim_kwargs = _construct_default_optim_kwargs(flag_ground_in_graph, edge_filters)

    logging.getLogger().info("Start retargeting from CMU Mocap to robot GR1 using graph distance")
    for k, v in optim_kwargs.items():
        logging.getLogger().info(f"{k}: {v}")
    logging.getLogger().info(f"normalize_pose: {normalize_pose}")
    logging.getLogger().info(f"root_dir: {root_dir}")
    logging.getLogger().info(f"index_file: {index_file}")
    logging.getLogger().info(f"asf_amc_root_dir: {asf_amc_root_dir}")
    logging.getLogger().info(f"motions to be retargeted: {motion_trajs}")
    logging.getLogger().info(f"key bodies for mocap data: {mocap_config.key_bodies}")
    logging.getLogger().info(f"urdf_file: {robot_config.name}")
    logging.getLogger().info(f"key joints for robot: {robot_config.key_joints}")
    logging.getLogger().info(f"index mapping for json file: {robot_config.index_mapping}")

    execute_retargeting_asf_amc(root_dir, file_index, motion_trajs, mocap_config, robot_config, optim_kwargs)


def main_AMASS_to_GR1():
    """
    Retarget AMASS motion capture data to the GR1 robot using graph distance optimization.
    """
    _setup_logger()

    root_dir = "./graph_distance_AMASS_to_GR1"
    os.makedirs(root_dir, exist_ok=True)

    # The pose used for normalization, should be one of "t_pose" and "stand"
    normalize_pose = "t_pose"

    # mocap config
    mocap_config = AMASS_DanceDB_SMPL_HG_WholeBody_Config(norm_pose_type=normalize_pose)

    # robot config
    robot_config = Fourier_GR1_WholeBody_Config(norm_pose_type=normalize_pose)

    assert (("ground" in mocap_config.key_bodies) and ("ground" in robot_config.key_joints)) or \
           (("ground" not in mocap_config.key_bodies) and ("ground" not in robot_config.key_joints))
    flag_ground_in_graph = (("ground" in mocap_config.key_bodies) and ("ground" in robot_config.key_joints))

    # edge filter that remove the edges between upper body and lower body
    edge_filters = []
    for start_idx in range(len(mocap_config.key_bodies)):
        for end_idx in range(start_idx + 1, len(mocap_config.key_bodies)):
            edge_filters.append([start_idx, end_idx])

    optim_kwargs = _construct_default_optim_kwargs(flag_ground_in_graph, edge_filters)

    logging.getLogger().info("Start retargeting from AMASS Mocap to GR1 robot using graph distance")
    for k, v in optim_kwargs.items():
        logging.getLogger().info(f"{k}: {v}")
    logging.getLogger().info(f"normalize_pose: {normalize_pose}")
    logging.getLogger().info(f"root_dir: {root_dir}")
    logging.getLogger().info(f"key bodies for mocap data: {mocap_config.key_bodies}")
    logging.getLogger().info(f"urdf_file: {robot_config.name}")
    logging.getLogger().info(f"key joints for robot: {robot_config.key_joints}")
    logging.getLogger().info(f"index mapping for json file: {robot_config.index_mapping}")

    bvh_root_dir = os.path.join(os.path.dirname(__file__), "mocap_data_converter", "rsc", "amass_bvh")

    execute_retargeting_bvh(root_dir, bvh_root_dir, mocap_config, robot_config, optim_kwargs)


def main_AMASS_to_Unitree_G1():
    """
    Retarget AMASS motion capture data to the GR1 robot using graph distance optimization.
    """
    _setup_logger()

    root_dir = "./graph_distance_AMASS_to_Unitree_G1"
    os.makedirs(root_dir, exist_ok=True)

    # The pose used for normalization, should be one of "t_pose" and "stand"
    normalize_pose = "t_pose"

    # mocap config
    mocap_config = AMASS_DanceDB_SMPL_HG_WholeBody_Config(norm_pose_type=normalize_pose)

    # robot config
    robot_config = Unitree_G1_WholeBody_Config(norm_pose_type=normalize_pose)

    assert (("ground" in mocap_config.key_bodies) and ("ground" in robot_config.key_joints)) or \
           (("ground" not in mocap_config.key_bodies) and ("ground" not in robot_config.key_joints))
    flag_ground_in_graph = (("ground" in mocap_config.key_bodies) and ("ground" in robot_config.key_joints))

    # edge filter that remove the edges between upper body and lower body
    edge_filters = []
    for start_idx in range(len(mocap_config.key_bodies)):
        for end_idx in range(start_idx + 1, len(mocap_config.key_bodies)):
            edge_filters.append([start_idx, end_idx])

    optim_kwargs = _construct_default_optim_kwargs(flag_ground_in_graph, edge_filters)

    logging.getLogger().info("Start retargeting from AMASS Mocap to GR1 robot using graph distance")
    for k, v in optim_kwargs.items():
        logging.getLogger().info(f"{k}: {v}")
    logging.getLogger().info(f"normalize_pose: {normalize_pose}")
    logging.getLogger().info(f"root_dir: {root_dir}")
    logging.getLogger().info(f"key bodies for mocap data: {mocap_config.key_bodies}")
    logging.getLogger().info(f"urdf_file: {robot_config.name}")
    logging.getLogger().info(f"key joints for robot: {robot_config.key_joints}")
    logging.getLogger().info(f"index mapping for json file: {robot_config.index_mapping}")

    bvh_root_dir = os.path.join(os.path.dirname(__file__), "mocap_data_converter", "rsc", "amass_bvh")

    execute_retargeting_bvh(root_dir, bvh_root_dir, mocap_config, robot_config, optim_kwargs)


def main_AMASS_to_Unitree_H1():
    """
    Retarget AMASS motion capture data to the GR1 robot using graph distance optimization.
    """
    _setup_logger()

    root_dir = "./graph_distance_AMASS_to_Unitree_H1"
    os.makedirs(root_dir, exist_ok=True)

    # The pose used for normalization, should be one of "t_pose" and "stand"
    normalize_pose = "t_pose"

    # mocap config
    mocap_config = AMASS_DanceDB_SMPL_HG_WholeBody_Config(norm_pose_type=normalize_pose)

    # robot config
    robot_config = Unitree_H1_WholeBody_Config(norm_pose_type=normalize_pose)

    assert (("ground" in mocap_config.key_bodies) and ("ground" in robot_config.key_joints)) or \
           (("ground" not in mocap_config.key_bodies) and ("ground" not in robot_config.key_joints))
    flag_ground_in_graph = (("ground" in mocap_config.key_bodies) and ("ground" in robot_config.key_joints))

    # edge filter that remove the edges between upper body and lower body
    edge_filters = []
    for start_idx in range(len(mocap_config.key_bodies)):
        for end_idx in range(start_idx + 1, len(mocap_config.key_bodies)):
            edge_filters.append([start_idx, end_idx])

    optim_kwargs = _construct_default_optim_kwargs(flag_ground_in_graph, edge_filters)

    logging.getLogger().info("Start retargeting from AMASS Mocap to GR1 robot using graph distance")
    for k, v in optim_kwargs.items():
        logging.getLogger().info(f"{k}: {v}")
    logging.getLogger().info(f"normalize_pose: {normalize_pose}")
    logging.getLogger().info(f"root_dir: {root_dir}")
    logging.getLogger().info(f"key bodies for mocap data: {mocap_config.key_bodies}")
    logging.getLogger().info(f"urdf_file: {robot_config.name}")
    logging.getLogger().info(f"key joints for robot: {robot_config.key_joints}")
    logging.getLogger().info(f"index mapping for json file: {robot_config.index_mapping}")

    bvh_root_dir = os.path.join(os.path.dirname(__file__), "mocap_data_converter", "rsc", "amass_bvh")

    execute_retargeting_bvh(root_dir, bvh_root_dir, mocap_config, robot_config, optim_kwargs)


def main_AMASS_to_Berkeley_Humanoid_Lite():
    """
    Retarget AMASS motion capture data to the GR1 robot using graph distance optimization.
    """
    _setup_logger()

    root_dir = "./graph_distance_AMASS_to_Berkeley_Humanoid_Lite"
    os.makedirs(root_dir, exist_ok=True)

    # The pose used for normalization, should be one of "t_pose" and "stand"
    normalize_pose = "t_pose"

    # mocap config
    mocap_config = AMASS_DanceDB_SMPL_HG_WholeBody_Config(norm_pose_type=normalize_pose)

    # robot config
    robot_config = Berkeley_Humanoid_Lite_WholeBody_Config(norm_pose_type=normalize_pose)

    assert (("ground" in mocap_config.key_bodies) and ("ground" in robot_config.key_joints)) or \
           (("ground" not in mocap_config.key_bodies) and ("ground" not in robot_config.key_joints))
    flag_ground_in_graph = (("ground" in mocap_config.key_bodies) and ("ground" in robot_config.key_joints))

    # edge filter that remove the edges between upper body and lower body
    edge_filters = []
    for start_idx in range(len(mocap_config.key_bodies)):
        for end_idx in range(start_idx + 1, len(mocap_config.key_bodies)):
            edge_filters.append([start_idx, end_idx])

    optim_kwargs = _construct_default_optim_kwargs(flag_ground_in_graph, edge_filters)

    logging.getLogger().info("Start retargeting from AMASS Mocap to GR1 robot using graph distance")
    for k, v in optim_kwargs.items():
        logging.getLogger().info(f"{k}: {v}")
    logging.getLogger().info(f"normalize_pose: {normalize_pose}")
    logging.getLogger().info(f"root_dir: {root_dir}")
    logging.getLogger().info(f"key bodies for mocap data: {mocap_config.key_bodies}")
    logging.getLogger().info(f"urdf_file: {robot_config.name}")
    logging.getLogger().info(f"key joints for robot: {robot_config.key_joints}")
    logging.getLogger().info(f"index mapping for json file: {robot_config.index_mapping}")

    bvh_root_dir = os.path.join(os.path.dirname(__file__), "mocap_data_converter", "rsc", "amass_bvh")

    execute_retargeting_bvh(root_dir, bvh_root_dir, mocap_config, robot_config, optim_kwargs)


def main_AMASS_to_Fourier_N1():
    """
    Retarget AMASS motion capture data to the GR1 robot using graph distance optimization.
    """
    _setup_logger()

    root_dir = "./graph_distance_AMASS_to_Fourier_N1"
    os.makedirs(root_dir, exist_ok=True)

    # The pose used for normalization, should be one of "t_pose" and "stand"
    normalize_pose = "t_pose"

    # mocap config
    mocap_config = AMASS_DanceDB_SMPL_HG_WholeBody_Config(norm_pose_type=normalize_pose)

    # robot config
    robot_config = Fourier_N1_WholeBody_Config(norm_pose_type=normalize_pose)

    assert (("ground" in mocap_config.key_bodies) and ("ground" in robot_config.key_joints)) or \
           (("ground" not in mocap_config.key_bodies) and ("ground" not in robot_config.key_joints))
    flag_ground_in_graph = (("ground" in mocap_config.key_bodies) and ("ground" in robot_config.key_joints))

    # edge filter that remove the edges between upper body and lower body
    edge_filters = []
    for start_idx in range(len(mocap_config.key_bodies)):
        for end_idx in range(start_idx + 1, len(mocap_config.key_bodies)):
            edge_filters.append([start_idx, end_idx])

    optim_kwargs = _construct_default_optim_kwargs(flag_ground_in_graph, edge_filters)

    logging.getLogger().info("Start retargeting from AMASS Mocap to GR1 robot using graph distance")
    for k, v in optim_kwargs.items():
        logging.getLogger().info(f"{k}: {v}")
    logging.getLogger().info(f"normalize_pose: {normalize_pose}")
    logging.getLogger().info(f"root_dir: {root_dir}")
    logging.getLogger().info(f"key bodies for mocap data: {mocap_config.key_bodies}")
    logging.getLogger().info(f"urdf_file: {robot_config.name}")
    logging.getLogger().info(f"key joints for robot: {robot_config.key_joints}")
    logging.getLogger().info(f"index mapping for json file: {robot_config.index_mapping}")

    bvh_root_dir = os.path.join(os.path.dirname(__file__), "mocap_data_converter", "rsc", "amass_bvh")

    execute_retargeting_bvh(root_dir, bvh_root_dir, mocap_config, robot_config, optim_kwargs)


def main_AMASS_to_Boston_Dynamics_Atlas_HD():
    """
    Retarget AMASS motion capture data to the GR1 robot using graph distance optimization.
    """
    _setup_logger()

    root_dir = "./graph_distance_AMASS_to_Boston_Dynamics_Atlas_HD"
    os.makedirs(root_dir, exist_ok=True)

    # The pose used for normalization, should be one of "t_pose" and "stand"
    normalize_pose = "t_pose"

    # mocap config
    mocap_config = AMASS_DanceDB_SMPL_HG_WholeBody_Config(norm_pose_type=normalize_pose)

    # robot config
    robot_config = Boston_Dynamics_Atlas_HD_WholeBody_Config(norm_pose_type=normalize_pose)

    assert (("ground" in mocap_config.key_bodies) and ("ground" in robot_config.key_joints)) or \
           (("ground" not in mocap_config.key_bodies) and ("ground" not in robot_config.key_joints))
    flag_ground_in_graph = (("ground" in mocap_config.key_bodies) and ("ground" in robot_config.key_joints))

    # edge filter that remove the edges between upper body and lower body
    edge_filters = []
    for start_idx in range(len(mocap_config.key_bodies)):
        for end_idx in range(start_idx + 1, len(mocap_config.key_bodies)):
            edge_filters.append([start_idx, end_idx])

    optim_kwargs = _construct_default_optim_kwargs(flag_ground_in_graph, edge_filters)

    logging.getLogger().info("Start retargeting from AMASS Mocap to GR1 robot using graph distance")
    for k, v in optim_kwargs.items():
        logging.getLogger().info(f"{k}: {v}")
    logging.getLogger().info(f"normalize_pose: {normalize_pose}")
    logging.getLogger().info(f"root_dir: {root_dir}")
    logging.getLogger().info(f"key bodies for mocap data: {mocap_config.key_bodies}")
    logging.getLogger().info(f"urdf_file: {robot_config.name}")
    logging.getLogger().info(f"key joints for robot: {robot_config.key_joints}")
    logging.getLogger().info(f"index mapping for json file: {robot_config.index_mapping}")

    bvh_root_dir = os.path.join(os.path.dirname(__file__), "mocap_data_converter", "rsc", "amass_bvh")

    execute_retargeting_bvh(root_dir, bvh_root_dir, mocap_config, robot_config, optim_kwargs)


def main_AMASS_to_Agibot_X1():
    """
    Retarget AMASS motion capture data to the GR1 robot using graph distance optimization.
    """
    _setup_logger()

    root_dir = "./graph_distance_AMASS_to_Agibot_X1"
    os.makedirs(root_dir, exist_ok=True)

    # The pose used for normalization, should be one of "t_pose" and "stand"
    normalize_pose = "t_pose"

    # mocap config
    mocap_config = AMASS_DanceDB_SMPL_HG_WholeBody_Config(norm_pose_type=normalize_pose)

    # robot config
    robot_config = Agibot_X1_WholeBody_Config(norm_pose_type=normalize_pose)

    assert (("ground" in mocap_config.key_bodies) and ("ground" in robot_config.key_joints)) or \
           (("ground" not in mocap_config.key_bodies) and ("ground" not in robot_config.key_joints))
    flag_ground_in_graph = (("ground" in mocap_config.key_bodies) and ("ground" in robot_config.key_joints))

    # edge filter that remove the edges between upper body and lower body
    edge_filters = []
    for start_idx in range(len(mocap_config.key_bodies)):
        for end_idx in range(start_idx + 1, len(mocap_config.key_bodies)):
            edge_filters.append([start_idx, end_idx])

    optim_kwargs = _construct_default_optim_kwargs(flag_ground_in_graph, edge_filters)

    logging.getLogger().info("Start retargeting from AMASS Mocap to GR1 robot using graph distance")
    for k, v in optim_kwargs.items():
        logging.getLogger().info(f"{k}: {v}")
    logging.getLogger().info(f"normalize_pose: {normalize_pose}")
    logging.getLogger().info(f"root_dir: {root_dir}")
    logging.getLogger().info(f"key bodies for mocap data: {mocap_config.key_bodies}")
    logging.getLogger().info(f"urdf_file: {robot_config.name}")
    logging.getLogger().info(f"key joints for robot: {robot_config.key_joints}")
    logging.getLogger().info(f"index mapping for json file: {robot_config.index_mapping}")

    bvh_root_dir = os.path.join(os.path.dirname(__file__), "mocap_data_converter", "rsc", "amass_bvh")

    execute_retargeting_bvh(root_dir, bvh_root_dir, mocap_config, robot_config, optim_kwargs)


def main_AMASS_to_iCub():
    """
    Retarget AMASS motion capture data to the GR1 robot using graph distance optimization.
    """
    _setup_logger()

    root_dir = "./graph_distance_AMASS_to_iCub"
    os.makedirs(root_dir, exist_ok=True)

    # The pose used for normalization, should be one of "t_pose" and "stand"
    normalize_pose = "t_pose"

    # mocap config
    mocap_config = AMASS_DanceDB_SMPL_HG_WholeBody_Config(norm_pose_type=normalize_pose)

    # robot config
    robot_config = iCub_WholeBody_Config(norm_pose_type=normalize_pose)

    assert (("ground" in mocap_config.key_bodies) and ("ground" in robot_config.key_joints)) or \
           (("ground" not in mocap_config.key_bodies) and ("ground" not in robot_config.key_joints))
    flag_ground_in_graph = (("ground" in mocap_config.key_bodies) and ("ground" in robot_config.key_joints))

    # edge filter that remove the edges between upper body and lower body
    edge_filters = []
    for start_idx in range(len(mocap_config.key_bodies)):
        for end_idx in range(start_idx + 1, len(mocap_config.key_bodies)):
            edge_filters.append([start_idx, end_idx])

    optim_kwargs = _construct_default_optim_kwargs(flag_ground_in_graph, edge_filters)

    logging.getLogger().info("Start retargeting from AMASS Mocap to GR1 robot using graph distance")
    for k, v in optim_kwargs.items():
        logging.getLogger().info(f"{k}: {v}")
    logging.getLogger().info(f"normalize_pose: {normalize_pose}")
    logging.getLogger().info(f"root_dir: {root_dir}")
    logging.getLogger().info(f"key bodies for mocap data: {mocap_config.key_bodies}")
    logging.getLogger().info(f"urdf_file: {robot_config.name}")
    logging.getLogger().info(f"key joints for robot: {robot_config.key_joints}")
    logging.getLogger().info(f"index mapping for json file: {robot_config.index_mapping}")

    bvh_root_dir = os.path.join(os.path.dirname(__file__), "mocap_data_converter", "rsc", "amass_bvh")

    execute_retargeting_bvh(root_dir, bvh_root_dir, mocap_config, robot_config, optim_kwargs)


if __name__ == "__main__":
    # Entrypoint for running the desired retargeting scenario.
    # Uncomment the desired main_* function to run.
    # main_CMU_to_GR1()
    main_AMASS_to_GR1()
    # main_AMASS_to_Variable_Sized_Robots()
    # main_AMASS_to_Variable_Sized_Robots_mp()
    # main_AMASS_to_Unitree_G1()
    # main_AMASS_to_Unitree_H1()
    # main_AMASS_to_Berkeley_Humanoid_Lite()
    # main_AMASS_to_Fourier_N1()
    # main_AMASS_to_Boston_Dynamics_Atlas_HD()
    # main_AMASS_to_Agibot_X1()
    # main_AMASS_to_iCub()
