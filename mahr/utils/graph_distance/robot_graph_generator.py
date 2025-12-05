import logging
import time
import os
from typing import Iterable, List, Tuple, Optional

import networkx as nx
import numpy as np
from PIL import Image

from mahr.config.RobotConfig import RobotConfig

class HeightMap:
    def __init__(self, height_file: Optional[str] = None):
        """Optional height map loader.

        If height_file is provided and loadable, a normalized grayscale height map is stored in
        self.height_data. Otherwise, self.height_data is set to None and get_height() will return 0.0
        as a safe fallback.
        """
        self.height_file = height_file
        self.radius_x = 20.0
        self.radius_y = 20.0
        self.elevation_z = 1.0

        self.height_data = None
        if height_file:
            try:
                img = Image.open(height_file).convert('L')  # convert to grey scale image
                height_arr = np.array(img, dtype=np.float32)
                height_arr = height_arr / 255.0  # normalize the height map to [0, 1]
                self.height_data = height_arr
            except Exception as e:
                logging.getLogger().warning(
                    f"Failed to load height map from '{height_file}': {e}. Falling back to flat ground (0.0)."
                )

    def get_height(self, x: float, y: float) -> float:
        # If no height data, default to flat ground at 0.0
        if self.height_data is None:
            return 0.0

        # Normalize x and y to [0, 1]
        norm_x = (x + self.radius_x) / (2 * self.radius_x)
        norm_y = (y + self.radius_y) / (2 * self.radius_y)

        # Map normalized coordinates to pixel indices
        img_height, img_width = self.height_data.shape[0], self.height_data.shape[1]
        pixel_x = int(norm_x * (img_width - 1))
        pixel_y = int((1 - norm_y) * (img_height - 1))  # Invert y-axis for image coordinates

        # Ensure pixel indices are within bounds
        pixel_x = int(np.clip(pixel_x, 0, img_width - 1))
        pixel_y = int(np.clip(pixel_y, 0, img_height - 1))

        # Retrieve height value from the height map
        height_value = float(self.height_data[pixel_y, pixel_x]) * self.elevation_z
        return float(height_value)


class RobotGraphCallable:
    """Callable object that generates robot vector graphs from joint angles."""

    def __init__(
        self,
        robot,
        robot_key_joints: List[str],
        edge_pairs: List[Tuple[int, int]],
        edge_norm_lookup: dict[frozenset[str], float],
        normalize_edge_weights: bool,
        height_map_path: Optional[str] = None,
    ):
        self.robot = robot
        self.robot_key_joints = robot_key_joints
        self.edge_pairs = edge_pairs
        self.edge_norm_lookup = edge_norm_lookup
        self.normalize_edge_weights = normalize_edge_weights
        # Initialize height map from provided path (or None -> flat fallback in HeightMap)
        self.height_map = HeightMap(height_map_path)

    def __call__(
        self,
        joint_angles: Iterable[float],
        root_rotation_matrix: np.ndarray,
        root_x_pos: float,
        root_y_pos: float,
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
                start_x_offset = float(root_rotation_matrix[0].dot(fk_current[end_joint][0:3, 3]))
                start_y_offset = float(root_rotation_matrix[1].dot(fk_current[end_joint][0:3, 3]))
                start_height = self.height_map.get_height(root_x_pos + start_x_offset, root_y_pos + start_y_offset)
                end_height = root_height + float(
                    root_rotation_matrix[2].dot(fk_current[end_joint][0:3, 3])
                )
                pos_vector = np.asarray([0.0, 0.0, end_height - start_height], dtype=float)
            elif end_joint == "ground":
                start_x_offset = float(root_rotation_matrix[0].dot(fk_current[start_joint][0:3, 3]))
                start_y_offset = float(root_rotation_matrix[1].dot(fk_current[start_joint][0:3, 3]))
                start_height = root_height + float(
                    root_rotation_matrix[2].dot(fk_current[start_joint][0:3, 3])
                )
                end_height = self.height_map.get_height(root_x_pos + start_x_offset, root_y_pos + start_y_offset)
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
    edge_filter: Iterable[Tuple[int, int]] | None = None,
    height_map_path: Optional[str] = None,
):
    logging.getLogger().info("Start generating graphs for robot")
    robot = robot_config.robot
    num_dofs = len(robot.actuated_joints)

    gc_lower_bounds = np.asarray(
        [robot.actuated_joints[idx].limit.lower for idx in range(len(robot.actuated_joint_names))])
    gc_upper_bounds = np.asarray(
        [robot.actuated_joints[idx].limit.upper for idx in range(len(robot.actuated_joint_names))])

    t_pose_fk = robot.link_fk(
        cfg=robot_config.norm_pose_in_rad,
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
    # Resolve height map path: prefer explicit argument, else environment variable MAHR_HEIGHT_MAP, else None
    resolved_height_map = height_map_path if height_map_path else os.environ.get("MAHR_HEIGHT_MAP")

    return (
        num_dofs,
        (gc_lower_bounds, gc_upper_bounds),
        t_pose_graph,
        RobotGraphCallable(
            robot,
            robot_key_joints,
            edge_pairs,
            edge_norm_lookup,
            normalize_edge_weights,
            height_map_path=resolved_height_map,
        ),
        t_pose_toes_horizontal_distance
    )
