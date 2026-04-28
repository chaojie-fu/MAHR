"""
Run either L-BFGS-B OR Differential Evolution for a single AMASS BVH frame
when retargeting to the Fourier GR1 robot using the same graph-distance
objective employed by motion_retarget_graph_distance.py.

Usage examples:
    # Run only L-BFGS-B
    python comparing_optimization_methods.py \
        --bvh /path/to/motion.bvh \
        --norm-pose-dir /path/to/norm_pose \
        --frame 0 \
        --optimizer lbfgsb \
        --cache-dir ./ComparingMethods

    # Run only Differential Evolution
    python comparing_optimization_methods.py \
        --bvh /path/to/motion.bvh \
        --norm-pose-dir /path/to/norm_pose \
        --frame 0 \
        --optimizer de \
        --cache-dir ./ComparingMethods

Notes:
- You must pick ONE optimizer via --optimizer {lbfgsb|de}.
- For L-BFGS-B a local bounded search around the midpoint is used.
- For Differential Evolution global bounds are used.
"""

import argparse
import json
import logging
import multiprocessing
import os
import pathlib
import sqlite3
import time
from pathlib import Path

import numpy as np
import scipy.optimize

from config.MocapConfig.AmassSmplHGConfig import AMASS_DanceDB_SMPL_HG_WholeBody_Config
from config.RobotConfig.FourierGR1Config import Fourier_GR1_WholeBody_Config
from mahr.utils.graph_distance.mocap_graph_generator import generate_vector_graphs_for_motion
from mahr.utils.graph_distance.robot_graph_generator import generate_vector_symbolic_graph_for_robot

# Reuse helpers to ensure identical behavior
from motion_retarget_graph_distance import (
    _setup_logger,
    _cached_motion_config,
    _construct_objective_function,
    _calculate_variable_bounds,
    _build_edge_filters,
)


def _parse_args():
    p = argparse.ArgumentParser(
        description="Compare L-BFGS-B vs Differential Evolution on one AMASS BVH frame for Fourier GR1.")
    p.add_argument("--bvh", required=True, help="Path to AMASS BVH motion file")
    p.add_argument("--norm-pose-dir", required=True,
                   help="Directory containing norm_pose_{t_pose|stand}.bvh for the same subject")
    p.add_argument("--frame", type=int, default=0,
                   help="Frame index to optimize (default: 0)")
    p.add_argument("--cache-dir", default="./ComparisonOfOptimizationMethods",
                   help="Output directory for cache (default: ./ComparisonOfOptimizationMethods)")
    p.add_argument("--norm-pose", choices=["t_pose", "stand"], default="t_pose",
                   help="Normalization pose used for both mocap and robot (default: t_pose)")
    p.add_argument("--db-path", default="./log/optimizations.sqlite",
                   help="Path to SQLite database to store optimization runs (default: ./log/optimizations.sqlite)")

    # Objective/graph toggles to mirror the main script
    p.add_argument("--no-weight-by-closeness", action="store_true",
                   help="Disable weighting by closeness in the objective")
    p.add_argument("--no-normalize-using-t-pose", action="store_true",
                   help="Disable edge weight normalization by T-pose")
    p.add_argument("--no-filter-edges", action="store_true",
                   help="Disable filtering edges when generating graphs")
    p.add_argument("--height-map", dest="height_map", type=str, default=None,
                   help="Path to grayscale PNG height map; if provided, use non-flat terrain")

    # Optimization parameters
    p.add_argument("--de-maxiter", type=int, default=1000, help="Differential Evolution max iterations")

    return p.parse_args()


def compare_single_frame_AMASS_to_GR1(
    bvh: str,
    norm_pose_dir: str,
    frame_idx: int,
    cache_dir: str,
    norm_pose: str,
    flag_weighted_optimization_by_closeness: bool,
    flag_normalize_edge_weights_using_T_pose: bool,
    flag_filter_edges_when_generating_graph: bool,
    height_map_png: str | None,
    optimizer: str,
    lbfgsb_maxiter: int,
    de_maxiter: int,
    de_popsize: int,
    de_strategy: str,
    de_tol: float,
    seed: int | None,
    db_path: str,
):
    logger = logging.getLogger()

    # Prepare directories
    os.makedirs(cache_dir, exist_ok=True)
    motion_dict_dir = os.path.join(cache_dir, "motion_dict")
    os.makedirs(motion_dict_dir, exist_ok=True)
    if flag_normalize_edge_weights_using_T_pose:
        graph_dir = os.path.join(cache_dir, "normalized_vector_graph_output")
    else:
        graph_dir = os.path.join(cache_dir, "vector_graph_output")
    os.makedirs(graph_dir, exist_ok=True)

    # Configs
    mocap_config = AMASS_DanceDB_SMPL_HG_WholeBody_Config(norm_pose_type=norm_pose)
    robot_config = Fourier_GR1_WholeBody_Config(norm_pose_type=norm_pose)
    edge_filters = _build_edge_filters(len(mocap_config.key_bodies))

    # Optional height map via env for downstream utilities
    if height_map_png:
        os.environ["MAHR_HEIGHT_MAP"] = height_map_png
        logger.info(f"Height map set to: {height_map_png}")
    else:
        if "MAHR_HEIGHT_MAP" in os.environ:
            del os.environ["MAHR_HEIGHT_MAP"]
        logger.info("No height map provided; using flat ground (0.0).")

    # Load motion and build graphs (single BVH)
    mocap_config = _cached_motion_config(cache_dir, mocap_config, bvh, norm_pose_dir)

    # Robot graph function and bounds
    (_, gc_bounds, robot_tpose_graph, robot_graph_fn, robot_t_pose_toes_horizontal_distance) = (
        generate_vector_symbolic_graph_for_robot(
            robot_config,
            normalize_edge_weights=flag_normalize_edge_weights_using_T_pose,
            filter_edges_when_generating_graph=flag_filter_edges_when_generating_graph,
            edge_filter=edge_filters,
            height_map_path=height_map_png,
        )
    )

    # Mocap graphs
    t_pose_graph, mocap_graphs, motion_global_dict, mocap_t_pose_toes_horizontal_distance = (
        generate_vector_graphs_for_motion(
            mocap_config,
            motion_dict_dir,
            graph_dir,
            normalize_edge_weights=flag_normalize_edge_weights_using_T_pose,
            force_generate_motion_dict=False,
            force_generate_graph=False,
            filter_edges_when_generating_graph=flag_filter_edges_when_generating_graph,
            edge_filter=edge_filters,
        )
    )

    num_frames = len(mocap_graphs)
    if frame_idx < 0 or frame_idx >= num_frames:
        raise IndexError(f"Frame index {frame_idx} out of range [0, {num_frames - 1}]")

    # Horizontal ratio for root plane translation
    horizontal_ratio = robot_t_pose_toes_horizontal_distance / mocap_t_pose_toes_horizontal_distance
    root_x_pos = float(motion_global_dict["global_translations"][frame_idx][0][0]) * horizontal_ratio
    root_y_pos = float(motion_global_dict["global_translations"][frame_idx][0][1]) * horizontal_ratio
    root_quat = motion_global_dict["global_rotations"][frame_idx][0]

    # Objective
    f = _construct_objective_function(
        mocap_graphs[frame_idx],
        mocap_config,
        robot_graph_fn,
        robot_config,
        root_quat,
        root_x_pos,
        root_y_pos,
        flag_weighted_optimization_by_closeness,
        flag_filter_edges_when_generating_graph,
        edge_filters,
    )

    # Initial guess and bounds (match motion_retarget_graph_distance defaults)
    variable_lower_bounds = np.concatenate([gc_bounds[0], np.asarray([0.0])], axis=0)
    variable_higher_bounds = np.concatenate([gc_bounds[1], np.asarray([4.0])], axis=0)
    x0 = variable_lower_bounds + (variable_higher_bounds - variable_lower_bounds) * 0.5

    def optimize_basis(x: np.ndarray) -> float:
        # Small joint penalty, same as main script
        return f(x) + 1e-3 * np.sum(np.square(np.clip(x, -0.78, 0.78)))

    # Evaluate baseline
    v0_pen = float(optimize_basis(x0))
    v0_raw = float(f(x0))

    bounds = _calculate_variable_bounds(gc_bounds, x0, constraint_x_by_x0=True)
    if optimizer == "lbfgsb":
        t0 = time.time()
        res_lbfgsb = scipy.optimize.minimize(
            optimize_basis,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": lbfgsb_maxiter, "maxls": 40},
        )
        t1 = time.time()
        x_opt = res_lbfgsb.x
    elif optimizer == "de":
        bounds = _calculate_variable_bounds(gc_bounds, x0, constraint_x_by_x0=False)
        t0 = time.time()
        res_de = scipy.optimize.differential_evolution(
            optimize_basis,
            bounds,
            init="sobol",
            strategy=de_strategy,
            maxiter=de_maxiter,
            popsize=de_popsize,
            tol=de_tol,
            seed=seed,
            polish=True,
        )
        t1 = time.time()
        x_opt = res_de.x
    else:
        raise ValueError(f"Unknown optimizer '{optimizer}'")
    # Compute final objective values and write to DB
    opt_pen = float(optimize_basis(x_opt))
    opt_raw = float(f(x_opt))
    runtime_sec = float(max(t1 - t0, 0.0))

    # DB helpers (mirrors style used in spatial_relationship_preservation_across_robots_motions_methods.py)
    def _ensure_parent_dir(path: str):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _open_db(path: str) -> sqlite3.Connection:
        _ensure_parent_dir(path)
        conn = sqlite3.connect(path, timeout=30.0)
        # conn.execute("PRAGMA journal_mode=WAL;")
        # conn.execute("PRAGMA synchronous=NORMAL;")
        # conn.execute("PRAGMA busy_timeout=30000;")
        return conn

    # Create table and upsert row
    conn = _open_db(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS optimization_runs_continuous (
            bvh_file_name TEXT NOT NULL,
            frame_index INTEGER NOT NULL,
            norm_pose TEXT NOT NULL,
            weight_by_closeness INTEGER NOT NULL,
            normalize_using_t_pose INTEGER NOT NULL,
            optimizer TEXT NOT NULL,

            -- Include ALL available optimizer arguments in the primary key per requirement
            lbfgsb_maxiter INTEGER NOT NULL,
            de_maxiter INTEGER NOT NULL,
            de_popsize INTEGER NOT NULL,
            de_strategy TEXT NOT NULL,
            de_tol REAL NOT NULL,

            -- Optimization results
            v0_raw REAL NOT NULL,
            v0_penalized REAL NOT NULL,
            opt_raw REAL NOT NULL,
            opt_penalized REAL NOT NULL,
            runtime_sec REAL NOT NULL,

            PRIMARY KEY (
                bvh_file_name,
                frame_index,
                norm_pose,
                weight_by_closeness,
                normalize_using_t_pose,
                optimizer,
                lbfgsb_maxiter,
                de_maxiter,
                de_popsize,
                de_strategy,
                de_tol
            )
        )
        """
    )

    UPSERT_SQL = """
    INSERT INTO optimization_runs_continuous (
        bvh_file_name, frame_index, norm_pose, weight_by_closeness, normalize_using_t_pose, optimizer,
        lbfgsb_maxiter, de_maxiter, de_popsize, de_strategy, de_tol,
        v0_raw, v0_penalized, opt_raw, opt_penalized, runtime_sec
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(
        bvh_file_name, frame_index, norm_pose, weight_by_closeness, normalize_using_t_pose, optimizer,
        lbfgsb_maxiter, de_maxiter, de_popsize, de_strategy, de_tol
    ) DO UPDATE SET
        v0_raw=excluded.v0_raw,
        v0_penalized=excluded.v0_penalized,
        opt_raw=excluded.opt_raw,
        opt_penalized=excluded.opt_penalized,
        runtime_sec=excluded.runtime_sec
    """

    # get bvh_file_name from bvh path. The bvh_file_name should be extensionless
    bvh_file_name = pathlib.Path(bvh).stem

    # Use sentinel values instead of NULLs for non-relevant optimizer parameters so that
    # the ON CONFLICT target never contains NULLs (in SQLite, NULLs in UNIQUE/PK do not conflict).
    SENTINEL_INT = -1
    SENTINEL_FLOAT = -1.0
    SENTINEL_TEXT = "N/A"

    lbfgsb_maxiter_val = int(lbfgsb_maxiter) if optimizer == "lbfgsb" else SENTINEL_INT
    de_maxiter_val = int(de_maxiter) if optimizer == "de" else SENTINEL_INT
    de_popsize_val = int(de_popsize) if optimizer == "de" else SENTINEL_INT
    de_strategy_val = str(de_strategy) if optimizer == "de" else SENTINEL_TEXT
    de_tol_val = float(de_tol) if optimizer == "de" else SENTINEL_FLOAT

    row = (
        bvh_file_name,
        int(frame_idx),
        str(norm_pose),
        1 if flag_weighted_optimization_by_closeness else 0,
        1 if flag_normalize_edge_weights_using_T_pose else 0,
        optimizer,
        lbfgsb_maxiter_val,
        de_maxiter_val,
        de_popsize_val,
        de_strategy_val,
        de_tol_val,
        float(v0_raw),
        float(v0_pen),
        float(opt_raw),
        float(opt_pen),
        runtime_sec,
    )
    cur = conn.cursor()
    cur.execute("BEGIN")
    cur.execute(UPSERT_SQL, row)
    conn.commit()
    conn.close()


def main():
    _setup_logger()
    args = _parse_args()

    compare_single_frame_AMASS_to_GR1(
        bvh=args.bvh,
        norm_pose_dir=args.norm_pose_dir,
        frame_idx=args.frame,
        cache_dir=args.cache_dir,
        norm_pose=args.norm_pose,
        flag_weighted_optimization_by_closeness=not args.no_weight_by_closeness,
        flag_normalize_edge_weights_using_T_pose=not args.no_normalize_using_t_pose,
        flag_filter_edges_when_generating_graph=not args.no_filter_edges,
        height_map_png=args.height_map,
        optimizer="de",
        lbfgsb_maxiter=5000,
        de_maxiter=args.de_maxiter,
        de_popsize=15,
        de_strategy="best1bin",
        de_tol=1e-4,
        seed=None,
        db_path=args.db_path
    )


if __name__ == "__main__":
    main()
