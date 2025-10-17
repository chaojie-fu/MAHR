import os
import sys

import pickle
from pathlib import Path

import sympy
import numpy as np
import networkx as nx
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FixedLocator
from scipy.ndimage.filters import gaussian_filter, median_filter
import multiprocessing
from scipy.stats import spearmanr
import scienceplots

from mahr.utils.graph_distance.mocap_graph_generator import generate_vector_graphs_for_motion
from mahr.utils.graph_distance.robot_graph_generator import generate_vector_symbolic_graph_for_robot
from mahr.config import MocapConfig, RobotConfig
from mahr.rsc import rsc_dir

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../config"))

from config.MocapConfig.AmassSmplHGConfig import AMASS_DanceDB_SMPL_HG_WholeBody_Config
from config.RobotConfig.VariableSizedHumanoidConfig import VariableSizedHumanoidConfig

matplotlib.use("TkAgg")
# plt.style.use('science')

# cache_root_dir = "graph_distance_AMASS_to_Different_Sized_Robots_custom_weighting"
# cache_root_dir = "graph_distance_AMASS_to_Different_Sized_Robots_coeff_10"
# cache_root_dir = "graph_distance_AMASS_to_Different_Sized_Robots_coeff_5"
cache_root_dir = "graph_distance_AMASS_to_Different_Sized_Robots_no_closeness"
# For multi-run comparisons, list multiple roots here
CACHE_ROOT_DIRS = [
    "graph_distance_AMASS_to_Different_Sized_Robots_coeff_5",
    "graph_distance_AMASS_to_Different_Sized_Robots_no_closeness"]
CACHE_ROOT_NAMES = [
    "w/ PW",
    "w/o PW"]
FONT_SIZE = 10
TITLE_SIZE = 12


# --------------------
# Helpers to improve readability/modularity
# --------------------

def prepare_ratio_axes():
    """Prepare arm/leg ratio grids and their display labels.

    Returns:
        arm_body_ratios, leg_body_ratios, arm_labels, leg_labels,
        robot_names, arm_leg_body_ratios, indexes
    """
    ratios = [1.0 + (i + 1) * 0.075 for i in range(10)]

    arm_body_ratios = ([rf"$\displaystyle\frac{{1}}{{ {ratio:.3f} }}$" for ratio in reversed(ratios)] +
                       [rf"$1.000$"] +
                       [rf"${ratio:.3f}$" for ratio in ratios])
    leg_body_ratios = ([rf"$1 / {ratio:.3f}$" for ratio in reversed(ratios)] +
                       [rf"$1.000$"] +
                       [rf"${ratio:.3f}$" for ratio in ratios])

    arm_labels = ([rf"$\displaystyle\frac{{1}}{{ {ratio:.2f} }}$" for ratio in reversed(ratios)] +
                  [rf"$1.00$"] +
                  [rf"${ratio:.2f}$" for ratio in ratios])
    leg_labels = ([rf"$1/{ratio:.2f}$" for ratio in reversed(ratios)] +
                  [rf"$1.00$"] +
                  [rf"${ratio:.2f}$" for ratio in ratios])

    robot_names = [f"{ab:02d}_{lb:02d}" for ab in range(21) for lb in range(21)]
    arm_leg_body_ratios = [(arm_body_ratios[ab], leg_body_ratios[lb]) for ab in range(21) for lb in range(21)]
    indexes = [(ab, lb) for ab in range(21) for lb in range(21)]
    return arm_body_ratios, leg_body_ratios, arm_labels, leg_labels, robot_names, arm_leg_body_ratios, indexes


def select_tick_positions(n: int, step: int = 4, preserve_middle: bool = True):
    """Select tick positions centered at the middle, stepping outward.

    Picks the middle index first, then adds indices at +/- k*step from the middle
    while they are within [0, n-1]. If n is even, the middle is defined as (n-1)//2.

    Args:
        n: total number of positions (0..n-1)
        step: stride for tick positions (must be >= 1)
        preserve_middle: whether to include the middle index as the first tick
    Returns:
        Sorted list of tick indices
    """
    if n <= 0:
        return []
    if step <= 0:
        step = 1
    mid = (n - 1) // 2

    ticks = []
    if preserve_middle:
        ticks.append(mid)

    k = 1
    while True:
        added = False
        right = mid + k * step
        left = mid - k * step
        if right < n:
            ticks.append(right)
            added = True
        if left >= 0:
            ticks.append(left)
            added = True
        if not added:
            break
        k += 1

    return sorted(set(ticks))


def load_robot_graphs_and_mapping(robot_names, arm_leg_body_ratios, indexes, mocap_frame_of_interest, cache_root):
    """Load robot graphs (from cache) and build the MultiwayMap mapping for a given cache_root.

    Returns:
        mapping (MultiwayMap), robot_graph_data (list[RobotGraphData])
    """
    args = [(name, mocap_frame_of_interest, cache_root) for name in robot_names]
    with multiprocessing.Pool() as pool:
        results = pool.map(process_robot, args)

    graph_distances = get_robot_graph_distance(robot_names, cache_root)

    robot_graph_data = []
    for idx in range(len(robot_names)):
        robot_graph_data.append(
            RobotGraphData(
                results[idx][1],
                indexes[idx],
                robot_names[idx],
                arm_leg_body_ratios[idx],
                graph_distances[idx]
            )
        )

    mapping = MultiwayMap()
    for item in robot_graph_data:
        mapping.add_object(item)
    return mapping, robot_graph_data


def compute_vectors_and_lengths(mocap_graph, mapping, edge_pairs, num_arms, num_legs):
    """Compute mocap edge vectors/lengths and retargeted ones for all robots.

    Returns:
        mocap_lengths [E], mocap_vecs [E,3], retargeted_lengths [A,L,E], retargeted_vecs [A,L,E,3]
    """
    num_edges = len(edge_pairs)

    mocap_lengths = np.zeros([num_edges])
    mocap_vecs = np.zeros([num_edges, 3])

    retargeted_lengths = np.zeros([num_arms, num_legs, num_edges])
    retargeted_vecs = np.zeros([num_arms, num_legs, num_edges, 3])

    example_robot_urdf = os.path.join(os.path.dirname(__file__), "robots", "parsed_humanoid", "00_00.urdf")
    mocap_key_bodies = AMASS_DanceDB_SMPL_HG_WholeBody_Config().key_bodies
    robot_key_joints = VariableSizedHumanoidConfig(example_robot_urdf).key_joints

    for edge_idx, (i, j) in enumerate(edge_pairs):
        mocap_ss = mocap_key_bodies[i]
        mocap_to = mocap_key_bodies[j]
        robot_ss = robot_key_joints[i]
        robot_to = robot_key_joints[j]

        mocap_pos_vector = mocap_graph.get_edge_data(mocap_ss, mocap_to)["position_vector"]
        mocap_vecs[edge_idx, :] = mocap_pos_vector[:]
        mocap_lengths[edge_idx] = np.linalg.norm(mocap_pos_vector)

        for arm_idx in range(num_arms):
            for leg_idx in range(num_legs):
                robot_data = mapping.get_by_key("index", (arm_idx, leg_idx))
                robot_graph = robot_data.graph
                robot_pos_vector = robot_graph.get_edge_data(robot_ss, robot_to)["position_vector"]
                retargeted_vecs[arm_idx, leg_idx, edge_idx, :] = robot_pos_vector[:]
                retargeted_lengths[arm_idx, leg_idx, edge_idx] = np.linalg.norm(robot_pos_vector)

    return mocap_lengths, mocap_vecs, retargeted_lengths, retargeted_vecs


def build_distance_grid(mapping, num_arms, num_legs):
    distance = np.zeros([num_arms, num_legs])
    for arm_idx in range(num_arms):
        for leg_idx in range(num_legs):
            robot_data = mapping.get_by_key("index", (arm_idx, leg_idx))
            distance[arm_idx, leg_idx] = robot_data.graph_distance
    return distance


def get_mocap_norm_pose_graph_and_pose_graph(
    robot_idx,
    norm_pose,
    normalize_edge_weights=True,
    filter_edges_when_generating_graph=True,
    cache_root=None
):
    # source data directories
    bvh_motion_dir = os.path.join(
        rsc_dir, "amass_bvh_spec")
    norm_pose_dir = os.path.join(rsc_dir, "amass_bvh_spec", "norm_pose")

    bvh_motion_file = os.path.join(bvh_motion_dir, "Stefanos_Zeibekiko_CY_C3D_poses_frame_3335.bvh")

    # use new cache dir rather than the one used for optimization
    _cache_dir, _motion_dict_dir, _graph_dir = get_cache_dirs(
        robot_idx,
        normalize_edge_weights,
        filter_edges_when_generating_graph,
        cache_root or cache_root_dir
    )
    _mocap_config = cached_motion_config(_cache_dir, norm_pose, bvh_motion_file, norm_pose_dir)
    edge_filters = get_edge_filters_for_whole_body()
    _mocap_t_pose_graph, _mocap_graphs, _, _ = (
        cached_motion_graph_generation(
            _cache_dir,
            _mocap_config,
            _motion_dict_dir,
            _graph_dir,
            normalize_edge_weights=normalize_edge_weights,
            force_generate_motion_dict=False,
            force_generate_graph=False,
            filter_edges_when_generating_graph=filter_edges_when_generating_graph,
            edge_filter=edge_filters))

    return _mocap_t_pose_graph, _mocap_graphs, _mocap_config, _cache_dir


def get_robot_norm_pose_graph_and_pose_graph(
    robot_idx,
    norm_pose,
    frame_of_interest,
    normalize_edge_weights=True,
    filter_edges_when_generating_graph=True,
    cache_root=None
):
    _cache_dir, _motion_dict_dir, _graph_dir = get_cache_dirs(
        robot_idx,
        normalize_edge_weights,
        filter_edges_when_generating_graph,
        cache_root or cache_root_dir
    )

    _robot_config = cached_robot_config(robot_idx, _cache_dir, norm_pose)
    optimized_cache_dir = os.path.join(
        os.path.dirname(__file__),
        f"../../../{cache_root or cache_root_dir}",
        f"{robot_idx}",
        "optimized_trajectories_vector_graph/Stefanos_Zeibekiko_CY_C3D_poses_frame_3335")
    optimized_cache_file = os.path.join(optimized_cache_dir, f"{frame_of_interest}.pkl")

    edge_filters = get_edge_filters_for_whole_body()

    with open(optimized_cache_file, "rb") as file:
        x0 = pickle.load(file)

    (gc_symbols, root_rot_symbols, root_height_symbol), _, _robot_t_pose_graph, _robot_symbolic_graph, _ = (
        cached_robot_graph_generation(
            _cache_dir,
            _robot_config,
            normalize_edge_weights=normalize_edge_weights,
            filter_edges_when_generating_graph=filter_edges_when_generating_graph
        ))

    _robot_graph = cached_robot_graph_substitution(
        _cache_dir,
        _robot_config,
        _robot_symbolic_graph,
        root_rot_symbols,
        gc_symbols,
        root_height_symbol,
        x0,
        normalize_edge_weights=normalize_edge_weights,
        filter_edges_when_generating_graph=filter_edges_when_generating_graph
    )

    return _robot_t_pose_graph, _robot_graph, _robot_config, _cache_dir


def get_robot_graph_distance(_robot_names, cache_root):
    graph_distances = np.zeros([len(_robot_names)])
    for idx, name in enumerate(_robot_names):
        optimized_cache_dir = os.path.join(
            os.path.dirname(__file__),
            f"../../../{cache_root}",
            f"{name}",
            "optimized_trajectories_vector_graph/Stefanos_Zeibekiko_CY_C3D_poses_frame_3335")
        graph_distance_file = os.path.join(optimized_cache_dir, "0_value.pkl")

        with open(graph_distance_file, "rb") as file:
            gd = pickle.load(file)

        # saved data is squared graph distance, compute square root before assigning
        graph_distances[idx] = np.sqrt(gd)

    return graph_distances


def get_cache_dirs(
    robot_idx,
    normalize_edge_weights_using_T_pose,
    filter_edges_when_generating_graph,
    cache_root
):
    # cache directories
    if filter_edges_when_generating_graph:
        _cache_dir = os.path.join(
            os.path.dirname(__file__),
            f"../../../{cache_root}/{robot_idx}/filter_edge"
        )
    else:
        _cache_dir = os.path.join(
            os.path.dirname(__file__),
            f"../../../{cache_root}/{robot_idx}/no_filter_edge"
        )
    os.makedirs(_cache_dir, exist_ok=True)

    _motion_dict_dir = os.path.join(_cache_dir, "motion_dict")
    os.makedirs(_motion_dict_dir, exist_ok=True)

    if normalize_edge_weights_using_T_pose:
        _graph_dir = os.path.join(_cache_dir, "normalized_vector_graph_output")
    else:
        _graph_dir = os.path.join(_cache_dir, "vector_graph_output")
    os.makedirs(_graph_dir, exist_ok=True)

    return _cache_dir, _motion_dict_dir, _graph_dir


def get_edge_filters_for_whole_body():
    _num_key_bodies = 15
    _edge_filters = []

    for start_idx in range(_num_key_bodies):
        for end_idx in range(start_idx + 1, _num_key_bodies):
            _edge_filters.append([start_idx, end_idx])
    for start_idx in [1, 2, 3, 4, 5, 6, 7]:
        for end_idx in [8, 9, 10, 11, 12, 13]:
            _edge_filters.remove([start_idx, end_idx])

    return _edge_filters


def get_edge_filters_for_visualization():
    _edge_filters = []

    # joints_of_interest = [0, 1, 2, 4, 5,  7, 9, 10, 12, 13]
    # for start_idx in range(len(joints_of_interest)):
    #     for end_idx in range(start_idx + 1, len(joints_of_interest)):
    #         _edge_filters.append([joints_of_interest[start_idx], joints_of_interest[end_idx]])

    # add all edges to edge_filter except for those connecting grounds
    for start_idx in range(14):
        for end_idx in range(start_idx + 1, 14):
            _edge_filters.append([start_idx, end_idx])

    return _edge_filters


def cached_motion_config(_cache_dir, _normalize_pose, _bvh_motion_file, _norm_pose_dir):
    cache_file = Path(os.path.join(
        _cache_dir,
        f"mc_cache_{_normalize_pose}.pkl"))

    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        _mocap_config = AMASS_DanceDB_SMPL_HG_WholeBody_Config(norm_pose_type=_normalize_pose)
        _mocap_config.load_motion(_bvh_motion_file, _norm_pose_dir)

        with open(cache_file, 'wb') as f:
            pickle.dump(_mocap_config, f)
        return _mocap_config


def cached_robot_config(robot_idx, _cache_dir, _normalize_pose):
    cache_file = Path(os.path.join(
        _cache_dir,
        f"rc_cache_{_normalize_pose}.pkl"))

    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        robot_urdf = os.path.join(os.path.dirname(__file__), "robots", "parsed_humanoid", f"{robot_idx}.urdf")
        _robot_config = VariableSizedHumanoidConfig(robot_urdf, norm_pose_type=_normalize_pose)

        with open(cache_file, 'wb') as f:
            pickle.dump(_robot_config, f)
        return _robot_config


def cached_motion_graph_generation(
    _cache_dir,
    _mocap_config: MocapConfig,
    _motion_dict_dir, _graph_dir,
    normalize_edge_weights=False,
    force_generate_motion_dict=True,
    force_generate_graph=True,
    filter_edges_when_generating_graph=True,
    edge_filter=None
):
    cache_file = Path(os.path.join(
        _cache_dir,
        f"mgg_cache_{normalize_edge_weights}_{filter_edges_when_generating_graph}.pkl"))
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        result = generate_vector_graphs_for_motion(
            _mocap_config,
            _motion_dict_dir,
            _graph_dir,
            normalize_edge_weights=normalize_edge_weights,
            force_generate_motion_dict=force_generate_motion_dict,
            force_generate_graph=force_generate_graph,
            filter_edges_when_generating_graph=filter_edges_when_generating_graph,
            edge_filter=edge_filter)
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        return result


def cached_robot_graph_generation(
    _cache_dir,
    _robot_config: RobotConfig,
    normalize_edge_weights=False,
    filter_edges_when_generating_graph=True,
    edge_filter=None
):
    cache_file = Path(os.path.join(
        _cache_dir,
        f"rgg_cache_{normalize_edge_weights}_{filter_edges_when_generating_graph}.pkl"))
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        result = generate_vector_symbolic_graph_for_robot(
            _robot_config,
            normalize_edge_weights=normalize_edge_weights,
            filter_edges_when_generating_graph=filter_edges_when_generating_graph,
            edge_filter=edge_filter)
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        return result


def cached_robot_graph_substitution(
    _cache_dir,
    _robot_config: RobotConfig,
    _robot_symbolic_graph,
    _root_rot_symbols,
    _gc_symbols,
    _root_height_symbol,
    _x0,
    normalize_edge_weights=False,
    filter_edges_when_generating_graph=True,
    edge_filter=None
):
    cache_file = Path(os.path.join(
        _cache_dir,
        f"rgs_cache_{normalize_edge_weights}_{filter_edges_when_generating_graph}.pkl"))
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        _robot_graph = nx.Graph()

        # manually set root rotation matrix according to the reference motion
        # as the graph of mocap is defined wrt root, any rotation matrix can be specified without changing the result
        root_mat = np.asarray([[1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0]])
        root_var_list = []
        root_value_list = []
        for row in range(3):
            for col in range(3):
                root_var_list.append(_root_rot_symbols[row][col])
                root_value_list.append(root_mat[row, col])

        graph_edge_props = {}
        for start_index, start_name in enumerate(_robot_config.key_joints):
            for end_index, end_name in enumerate(_robot_config.key_joints):
                edge_data = _robot_symbolic_graph.get_edge_data(start_name, end_name)
                if edge_data is not None:
                    position_vector_subs = np.zeros([3, ])
                    for idx in range(3):
                        f = sympy.lambdify(
                            _gc_symbols + [_root_height_symbol] + root_var_list,
                            edge_data["position_vector"][idx]
                        )
                        position_vector_subs[idx] = f(*(_x0.tolist() + root_value_list))

                    graph_edge_props[(start_name, end_name)] = {
                        "position_vector": position_vector_subs
                    }
        _robot_graph.add_edges_from(graph_edge_props.keys())
        nx.set_edge_attributes(_robot_graph, graph_edge_props)

        with open(cache_file, 'wb') as f:
            pickle.dump(_robot_graph, f)
        return _robot_graph


def process_robot(_args):
    _robot_name, _mocap_frame_of_interest, _cache_root = _args
    print(f"Loading robot: {_robot_name} | cache: {_cache_root}")

    _robot_t_pose_graph, _robot_graph, _robot_config, _cache_dir_robot = get_robot_norm_pose_graph_and_pose_graph(
        _robot_name,
        "t_pose",
        _mocap_frame_of_interest,
        normalize_edge_weights=False,
        filter_edges_when_generating_graph=False,
        cache_root=_cache_root
    )

    return _robot_name, _robot_graph


def mark_max_value(axis, X_grid, Y_grid, Z_value, marker="*"):
    max_index = np.unravel_index(np.argmax(Z_value), Z_value.shape)
    i, j = max_index
    x_center = X_grid[i, j]
    y_center = Y_grid[i, j]
    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()

    dx = 0.06 * (xmax - xmin)
    dy = 0.06 * (ymax - ymin)

    if x_center > (xmax - 0.1*(xmax - xmin)):
        ha = 'right'
        dx = -dx  # 向左偏移
    else:
        ha = 'left'

    if y_center > (ymax - 0.1*(ymax - ymin)):
        va = 'top'
        dy = -dy
    else:
        va = 'bottom'
    axis.scatter(x_center, y_center, color='black', marker=marker, s=40, label='Max Value')
    axis.text(x_center + dx, y_center + dy, f'Max: {Z_value[i, j]:.2f}',
              ha=ha, va=va, color='black', fontsize=FONT_SIZE,
              bbox=dict(
                  facecolor="white",
                  edgecolor="none",
                  alpha=0.7
              )
              )


def mark_min_value(axis, X_grid, Y_grid, Z_value, marker="*"):
    min_index = np.unravel_index(np.argmin(Z_value), Z_value.shape)
    i, j = min_index
    x_center = X_grid[i, j]
    y_center = Y_grid[i, j]
    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()

    dx = 0.06 * (xmax - xmin)
    dy = 0.06 * (ymax - ymin)

    if x_center > (xmax - 0.5*(xmax - xmin)):
        ha = 'right'
        dx = -dx  # 向左偏移
    else:
        ha = 'left'

    if y_center > (ymax - 0.5*(ymax - ymin)):
        va = 'top'
        dy = -dy
    else:
        va = 'bottom'
    axis.scatter(x_center, y_center, color='black', marker=marker, s=40, label='Min Value')
    axis.text(x_center + dx, y_center + dy, f'Min: {Z_value[i, j]:.2f}',
              ha=ha, va=va, color='black', fontsize=FONT_SIZE,
              bbox=dict(
                  facecolor="white",
                  edgecolor="none",
                  alpha=0.7
              )
              )


class RobotGraphData:
    def __init__(self, graph, index, name, body_ratio, graph_distance):
        self.graph = graph
        self.index = index
        self.name = name
        self.body_ratio = body_ratio
        self.graph_distance = graph_distance


class MultiwayMap:
    def __init__(self):
        self.obj_store = {}  # {id: object}
        self.index_maps = {
            'index': {},  # {id_value: obj_id}
            'name': {},  # {name: obj_id}
        }
        self.linked_attrs = ['index', 'name']  # 需要建立索引的属性

    def add_object(self, obj):
        self.obj_store[obj.index] = obj

        # check if any one of the linked_attrs has been set
        for attr in self.linked_attrs:
            key = getattr(obj, attr)
            if key in self.index_maps[attr]:
                raise ValueError(f"{attr} {key} exists")

        # assign mapping after all the checks passed
        for attr in self.linked_attrs:
            key = getattr(obj, attr)
            self.index_maps[attr][key] = obj.index

    def get_by_key(self, attr, value):
        obj_id = self.index_maps[attr].get(value)
        return self.obj_store.get(obj_id) if obj_id else None


def main():
    # Choose where to save comparison figures (independent of specific cache roots)
    visualization_dir = os.path.join(
        os.path.dirname(__file__),
        f"../../../visualization_compare_multiple"
    )
    os.makedirs(visualization_dir, exist_ok=True)

    mocap_frame_of_interest = 0

    # Prepare ratios, robot names and indices
    arm_body_ratios, leg_body_ratios, arm_body_ratios_labels, leg_body_ratios_labels, robot_names, \
        arm_leg_body_ratios, indexes = prepare_ratio_axes()

    # We will compute and store per-cache-root grids
    results_per_root = []  # list of dicts: {"label": str, "spearman": arr, "distance": arr}

    # Build mocap graphs once per cache root (for cache separation); graphs are identical across roots
    for root in CACHE_ROOT_DIRS:
        # mocap graph for frame
        mocap_t_pose_graph, mocap_graphs, mocap_config, cache_dir_mocap = get_mocap_norm_pose_graph_and_pose_graph(
            "00_00",
            "t_pose",
            normalize_edge_weights=False,
            filter_edges_when_generating_graph=False,
            cache_root=root
        )
        mocap_graph = mocap_graphs[mocap_frame_of_interest]

        # Load robot graphs and construct mapping for this root
        mapping, robot_graph_data = load_robot_graphs_and_mapping(
            robot_names, arm_leg_body_ratios, indexes, mocap_frame_of_interest, root
        )

        # Compute vectors and distances for visualization
        edge_filtered = get_edge_filters_for_visualization()
        num_arms, num_legs = len(arm_body_ratios), len(leg_body_ratios)
        mocap_lengths, mocap_vecs, retargeted_lengths, retargeted_vecs = compute_vectors_and_lengths(
            mocap_graph, mapping, edge_filtered, num_arms, num_legs
        )

        distance = build_distance_grid(mapping, num_arms, num_legs)

        # Spearman rho per grid cell
        spearmanrs = np.zeros([num_arms, num_legs])
        for arm_idx in range(num_arms):
            for leg_idx in range(num_legs):
                spearmanrs[arm_idx, leg_idx] = spearmanr(
                    mocap_lengths, retargeted_lengths[arm_idx, leg_idx, :]
                ).statistic

        # Apply small denoise for visuals (same as original)
        spearmanrs = median_filter(spearmanrs, size=3)
        distance_smoothed = median_filter(distance, size=3)

        results_per_root.append({
            "label": root,
            "spearman": spearmanrs,
            "distance": distance_smoothed
        })

    # Determine global color scales for fair comparison
    if results_per_root:
        pass
    else:
        pass

    with plt.style.context('science'):
        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = "Arial"
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["font.size"] = 6
        plt.rcParams["text.latex.preamble"] = r"""
            \usepackage{sfmath}
            \usepackage{helvet}
            \renewcommand{\familydefault}{\rmdefault}
            """

        K = max(1, len(results_per_root))
        fig, axes = plt.subplots(2, K, figsize=(6.0, 6.0), dpi=600, squeeze=False)

        X = np.arange(len(arm_body_ratios))
        Y = np.arange(len(leg_body_ratios))
        X, Y = np.meshgrid(X, Y, indexing="ij")

        # compute reduced tick positions and labels (every 4, preserve middle)
        x_ticks = select_tick_positions(len(arm_body_ratios), step=4, preserve_middle=True)
        y_ticks = select_tick_positions(len(leg_body_ratios), step=4, preserve_middle=True)
        x_tick_labels = [arm_body_ratios_labels[i] for i in x_ticks]
        y_tick_labels = [leg_body_ratios_labels[i] for i in y_ticks]

        for col, res in enumerate(results_per_root):
            # Top row: Spearman
            ax_top = axes[0, col]
            m1 = ax_top.pcolormesh(
                X, Y, res["spearman"], cmap=cm.coolwarm_r, shading="gouraud",
                vmin=float(res["spearman"].min()), vmax=float(res["spearman"].max())
            )
            ax_top.contour(X, Y, gaussian_filter(res["spearman"], sigma=1),
                           colors="black", alpha=1.0, linewidths=1.0)
            ax_top.set_xlim(-0.5, len(arm_body_ratios) - 0.5)
            ax_top.set_ylim(-0.5, len(leg_body_ratios) - 0.5)
            # set major ticks every 4 with middle preserved
            ax_top.xaxis.set_major_locator(FixedLocator(x_ticks))
            ax_top.xaxis.set_minor_locator(FixedLocator(range(0, len(arm_body_ratios), 1)))
            ax_top.yaxis.set_major_locator(FixedLocator(y_ticks))
            ax_top.yaxis.set_minor_locator(FixedLocator(range(0, len(leg_body_ratios), 1)))
            mark_max_value(ax_top, X, Y, res["spearman"], "*")
            mark_min_value(ax_top, X, Y, res["spearman"], "x")
            if col == 0:
                ax_top.set_ylabel("Leg Length / Body Length", fontsize=FONT_SIZE)
            ax_top.set_title(f"Order-Preserving {CACHE_ROOT_NAMES[col]}", fontsize=TITLE_SIZE)
            ax_top.set_aspect('equal', adjustable='box')
            ax_top.tick_params(axis="both", direction="out", which="both", top=False, right=False)
            ax_top.tick_params(axis="x", which="major", pad=15)
            # Only bottom row should show x tick labels; hide on top row using tick_params
            ax_top.tick_params(labelbottom=False)
            # Only the leftmost column should show y tick labels
            if col == 0:
                ax_top.set_yticklabels(y_tick_labels, va='baseline', fontsize=FONT_SIZE)
            else:
                ax_top.tick_params(labelleft=False)
            # separate colorbar for the top subplot
            cbar = fig.colorbar(m1, ax=ax_top, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=FONT_SIZE)

            # Bottom row: Distance
            ax_bottom = axes[1, col]
            m2 = ax_bottom.pcolormesh(
                X, Y, res["distance"], cmap=cm.GnBu_r, shading="gouraud",
                vmin=float(res["distance"].min()), vmax=float(res["distance"].max())
            )
            ax_bottom.contour(X, Y, gaussian_filter(res["distance"], sigma=2),
                              colors="black", alpha=1.0, linewidths=1.0)
            ax_bottom.set_xlim(-0.5, len(arm_body_ratios) - 0.5)
            ax_bottom.set_ylim(-0.5, len(leg_body_ratios) - 0.5)
            # set major ticks every 4 with middle preserved
            ax_bottom.xaxis.set_major_locator(FixedLocator(x_ticks))
            ax_bottom.xaxis.set_minor_locator(FixedLocator(range(0, len(arm_body_ratios), 1)))
            ax_bottom.yaxis.set_major_locator(FixedLocator(y_ticks))
            ax_bottom.yaxis.set_minor_locator(FixedLocator(range(0, len(leg_body_ratios), 1)))
            mark_max_value(ax_bottom, X, Y, res["distance"], "x")
            mark_min_value(ax_bottom, X, Y, res["distance"], "*")
            ax_bottom.set_xlabel("Arm Length / Body Length", fontsize=FONT_SIZE)
            if col == 0:
                ax_bottom.set_ylabel("Leg Length / Body Length", fontsize=FONT_SIZE)
            ax_bottom.set_title(f"Graph Distance {CACHE_ROOT_NAMES[col]}", fontsize=TITLE_SIZE)
            ax_bottom.set_aspect('equal', adjustable='box')
            ax_bottom.tick_params(axis="both", direction="out", which="both", top=False, right=False)
            ax_bottom.tick_params(axis="x", which="major", pad=15)
            # Bottom row shows x tick labels for all columns
            ax_bottom.set_xticklabels(x_tick_labels, va='baseline', fontsize=FONT_SIZE)
            # Only the leftmost column should show y tick labels
            if col == 0:
                ax_bottom.set_yticklabels(y_tick_labels, va='baseline', fontsize=FONT_SIZE)
            else:
                ax_bottom.tick_params(labelleft=False)
            # separate colorbar for the bottom subplot
            cbar = fig.colorbar(m2, ax=ax_bottom, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=FONT_SIZE)

        # remove shared colorbars (replaced by per-subplot)
        # fig.set_tight_layout(True)
        fig.subplots_adjust(wspace=0.3, hspace=0.00)

        label_stub = "__".join([Path(r["label"]).name for r in results_per_root])
        out_name = f"Compare_Spearman_and_Distance_{label_stub}.png"
        fig.savefig(os.path.join(visualization_dir, out_name))

        # Optional: plt.show()


if __name__ == "__main__":
    main()
