import logging
from typing import Iterable, List, Tuple

import networkx as nx
import numpy as np

from mocap_data_converter.config.RobotConfig import RobotConfig


class RobotGraphCallable:
    """Callable object that generates robot vector graphs from joint angles."""

    def __init__(
        self,
        robot,
        robot_key_joints: List[str],
        edge_pairs: List[Tuple[int, int]],
        edge_norm_lookup: dict[frozenset[str], float],
        normalize_edge_weights: bool,
    ):
        self.robot = robot
        self.robot_key_joints = robot_key_joints
        self.edge_pairs = edge_pairs
        self.edge_norm_lookup = edge_norm_lookup
        self.normalize_edge_weights = normalize_edge_weights

    def __call__(
        self,
        joint_angles: Iterable[float],
        root_rotation_matrix: np.ndarray,
        root_height: float
    ) -> nx.Graph:
        cfg_dict = {
            joint_name: float(joint_value)
            for joint_name, joint_value in zip(self.robot.actuated_joint_names, joint_angles)
        }
        fk_current = self.robot.link_fk(cfg=cfg_dict, use_names=True)

        graph = nx.Graph()
        graph.add_nodes_from(self.robot_key_joints)

        vector_edges_runtime = []
        for start_idx_, end_idx_ in self.edge_pairs:
            start_joint = self.robot_key_joints[start_idx_]
            end_joint = self.robot_key_joints[end_idx_]

            if start_joint == "ground":
                start_height = 0.0
                end_height = root_height + float(
                    root_rotation_matrix[2].dot(fk_current[end_joint][0:3, 3])
                )
                pos_vector = np.asarray([0.0, 0.0, end_height - start_height], dtype=float)
            elif end_joint == "ground":
                start_height = root_height + float(
                    root_rotation_matrix[2].dot(fk_current[start_joint][0:3, 3])
                )
                end_height = 0.0
                pos_vector = np.asarray([0.0, 0.0, end_height - start_height], dtype=float)
            else:
                pos_vector = (
                    fk_current[end_joint][0:3, 3] -
                    fk_current[start_joint][0:3, 3]
                ).astype(float)

            if self.normalize_edge_weights:
                norm = self.edge_norm_lookup[frozenset((start_joint, end_joint))]
                if norm > 0.0:
                    pos_vector = pos_vector / norm

            vector_edges_runtime.append(
                (start_joint, end_joint, {"position_vector": pos_vector})
            )

        graph.add_edges_from(vector_edges_runtime)
        return graph


def generate_vector_symbolic_graph_for_robot(
    robot_config: RobotConfig,
    normalize_edge_weights: bool = False,
    filter_edges_when_generating_graph: bool = False,
    edge_filter: Iterable[Tuple[int, int]] | None = None
):
    logging.getLogger().info("Start generating graphs for robot")
    robot = robot_config.robot
    num_dofs = len(robot.actuated_joints)

    gc_lower_bounds = np.asarray(
        [robot.actuated_joints[idx].limit.lower for idx in range(len(robot.actuated_joint_names))])
    gc_upper_bounds = np.asarray(
        [robot.actuated_joints[idx].limit.upper for idx in range(len(robot.actuated_joint_names))])

    t_pose_fk = robot.link_fk(
        cfg=dict(zip(robot.actuated_joint_names, robot_config.norm_pose_in_rad)),
        use_names=True
    )

    toes_translations = [t_pose_fk[toe][0:3, 3] for toe in robot_config.toes]
    t_pose_toes_horizontal_distance = np.linalg.norm(toes_translations[0][0:2] - toes_translations[1][0:2], ord=2)

    # ====================================================================================================
    # t-pose graph
    robot_key_joints = robot_config.key_joints

    t_pose_graph = nx.Graph()
    t_pose_graph.add_nodes_from(robot_key_joints)

    # compute root elevations so that the toes are on the ground
    toe_heights = []
    for toe_name in robot_config.toes:
        toe_heights.append(float(t_pose_fk[toe_name][2, 3]))

    # root height of the current t-pose relative to desired on-ground height
    # equivalent to toe height of the current t-pose relative to desired on-ground height
    root_elevation = min(toe_heights) * 1.05

    if filter_edges_when_generating_graph:
        if edge_filter is None:
            raise ValueError("`edge_filter` must be provided when filtering edges.")
        edge_pairs: List[Tuple[int, int]] = [tuple(pair) for pair in edge_filter]
    else:
        edge_pairs = [
            (start_idx_, end_idx_)
            for start_idx_ in range(len(robot_key_joints))
            for end_idx_ in range(start_idx_ + 1, len(robot_key_joints))
        ]

    vector_edges = []
    edge_norm_lookup: dict[frozenset[str], float] = {}
    for start_idx_, end_idx_ in edge_pairs:
        start_joint = robot_key_joints[start_idx_]
        end_joint = robot_key_joints[end_idx_]
        if start_joint == "ground":
            start_height = 0.0
            end_height = t_pose_fk[end_joint][2, 3] - root_elevation
            pos_vector = np.asarray([0.0, 0.0, end_height - start_height], dtype=float)
        elif end_joint == "ground":
            start_height = t_pose_fk[start_joint][2, 3] - root_elevation
            end_height = 0.0
            pos_vector = np.asarray([0.0, 0.0, end_height - start_height], dtype=float)
        else:
            pos_vector = (
                t_pose_fk[end_joint][0:3, 3] -
                t_pose_fk[start_joint][0:3, 3]
            ).astype(float)

        vector_edges.append((start_joint, end_joint, {"position_vector": pos_vector}))
        edge_norm_lookup[frozenset((start_joint, end_joint))] = float(np.linalg.norm(pos_vector))

    t_pose_graph.add_edges_from(vector_edges)
    # ====================================================================================================

    # ====================================================================================================
    # robot graph
    return (
        num_dofs,
        (gc_lower_bounds, gc_upper_bounds),
        t_pose_graph,
        RobotGraphCallable(robot, robot_key_joints, edge_pairs, edge_norm_lookup, normalize_edge_weights),
        t_pose_toes_horizontal_distance
    )
