import logging

import networkx as nx
import numpy as np
import sympy
from urdfpy import URDF

from mocap_data_converter.config.RobotConfig import RobotConfig


def generate_vector_symbolic_graph_for_robot(
    robot_config: RobotConfig,
    normalize_edge_weights=False,
    filter_edges_when_generating_graph=False,
    edge_filter=None
):
    logging.getLogger().info("Start generating graphs for robot")
    robot = robot_config.robot
    num_dofs = len(robot.actuated_joints)

    gc_symbols = [sympy.symbols('p%d' % i) for i in range(num_dofs)]

    # naming format: r_row_col
    root_rot_symbols = [[sympy.symbols(f"r_{row}_{col}") for col in range(3)] for row in range(3)]
    root_height_symbol = sympy.symbols("h")

    fk = robot.link_fk(cfg=gc_symbols, use_names=True)

    gc_lower_bounds = np.asarray(
        [robot.actuated_joints[idx].limit.lower for idx in range(len(robot.actuated_joint_names))])
    gc_upper_bounds = np.asarray(
        [robot.actuated_joints[idx].limit.upper for idx in range(len(robot.actuated_joint_names))])

    t_pose_fk = robot.link_fk(cfg=robot_config.norm_pose_in_rad, use_names=True)

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

    vector_edges = []
    if filter_edges_when_generating_graph:
        assert edge_filter is not None
        for start_idx_, end_idx_ in edge_filter:
            if robot_key_joints[start_idx_] == "ground":
                start_height = 0.0
                # end_height = t_pose_fk[robot_key_joints[end_idx_]][2, 3] - root_elevation
                # use root height for edge normalization connected with ground
                end_height = t_pose_fk[robot_key_joints[end_idx_]][2, 3] - root_elevation
                props = {"position_vector": np.asarray([0.0, 0.0, end_height - start_height])}
            elif robot_key_joints[end_idx_] == "ground":
                # start_height = t_pose_fk[robot_key_joints[start_idx_]][2, 3] - root_elevation
                start_height = t_pose_fk[robot_key_joints[start_idx_]][2, 3] - root_elevation
                end_height = 0.0
                props = {"position_vector": np.asarray([0.0, 0.0, end_height - start_height])}
            else:
                props = {
                    "position_vector": t_pose_fk[robot_key_joints[end_idx_]][0:3, 3] -
                                       t_pose_fk[robot_key_joints[start_idx_]][0:3, 3]
                }
            vector_edges.append((robot_key_joints[start_idx_], robot_key_joints[end_idx_], props))
    else:
        for start_idx_ in range(len(robot_key_joints)):
            for end_idx_ in range(start_idx_ + 1, len(robot_key_joints)):
                if robot_key_joints[start_idx_] == "ground":
                    start_height = 0.0
                    # end_height = t_pose_fk[robot_key_joints[end_idx_]][2, 3] - root_elevation
                    end_height = t_pose_fk[robot_key_joints[end_idx_]][2, 3] - root_elevation
                    props = {"position_vector": np.asarray([0.0, 0.0, end_height - start_height])}
                elif robot_key_joints[end_idx_] == "ground":
                    # start_height = t_pose_fk[robot_key_joints[start_idx_]][2, 3] - root_elevation
                    start_height = t_pose_fk[robot_key_joints[start_idx_]][2, 3] - root_elevation
                    end_height = 0.0
                    props = {"position_vector": np.asarray([0.0, 0.0, end_height - start_height])}
                else:
                    props = {
                        "position_vector": t_pose_fk[robot_key_joints[end_idx_]][0:3, 3] -
                                           t_pose_fk[robot_key_joints[start_idx_]][0:3, 3]
                    }
                vector_edges.append((robot_key_joints[start_idx_], robot_key_joints[end_idx_], props))
    t_pose_graph.add_edges_from(vector_edges)
    # ====================================================================================================

    # ====================================================================================================
    # robot graph
    symbolic_graph = nx.Graph()
    symbolic_graph.add_nodes_from(robot_key_joints)

    vector_edges = []
    if filter_edges_when_generating_graph:
        assert edge_filter is not None
        for start_idx_, end_idx_ in edge_filter:
            if robot_key_joints[start_idx_] == "ground":
                start_height = 0.0
                end_height = root_height_symbol + \
                             np.dot(
                                 np.asarray([root_rot_symbols[2][0], root_rot_symbols[2][1], root_rot_symbols[2][2]]),
                                 fk[robot_key_joints[end_idx_]][0:3, 3]
                             )
                pos_vector = np.asarray([0.0, 0.0, end_height - start_height], dtype="O")
            elif robot_key_joints[end_idx_] == "ground":
                start_height = root_height_symbol + \
                               np.dot(
                                   np.asarray([root_rot_symbols[2][0], root_rot_symbols[2][1], root_rot_symbols[2][2]]),
                                   fk[robot_key_joints[start_idx_]][0:3, 3]
                               )
                end_height = 0.0
                pos_vector = np.asarray([0.0, 0.0, end_height - start_height], dtype="O")
            else:
                pos_vector = np.asarray(
                    fk[robot_key_joints[end_idx_]][0:3, 3] -
                    fk[robot_key_joints[start_idx_]][0:3, 3], dtype="O")

            if normalize_edge_weights:
                norm_vec = t_pose_graph.get_edge_data(
                    robot_key_joints[start_idx_],
                    robot_key_joints[end_idx_])["position_vector"]
                pos_vector /= sympy.sqrt(norm_vec[0] ** 2 + norm_vec[1] ** 2 + norm_vec[2] ** 2)

            props = {"position_vector": pos_vector}

            vector_edges.append((robot_key_joints[start_idx_], robot_key_joints[end_idx_], props))
    else:
        for start_idx_ in range(len(robot_key_joints)):
            for end_idx_ in range(start_idx_ + 1, len(robot_key_joints)):
                if robot_key_joints[start_idx_] == "ground":
                    start_height = 0.0
                    end_height = root_height_symbol + \
                                 np.dot(
                                     np.asarray([root_rot_symbols[2][0], root_rot_symbols[2][1],
                                                 root_rot_symbols[2][2]]),
                                     fk[robot_key_joints[end_idx_]][0:3, 3]
                                 )
                    pos_vector = np.asarray([0.0, 0.0, end_height - start_height], dtype="O")
                elif robot_key_joints[end_idx_] == "ground":
                    start_height = root_height_symbol + \
                                   np.dot(
                                       np.asarray([root_rot_symbols[2][0], root_rot_symbols[2][1],
                                                   root_rot_symbols[2][2]]),
                                       fk[robot_key_joints[start_idx_]][0:3, 3]
                                   )
                    end_height = 0.0
                    pos_vector = np.asarray([0.0, 0.0, end_height - start_height], dtype="O")
                else:
                    pos_vector = np.asarray(
                        fk[robot_key_joints[end_idx_]][0:3, 3] -
                        fk[robot_key_joints[start_idx_]][0:3, 3], dtype="O")

                if normalize_edge_weights:
                    norm_vec = t_pose_graph.get_edge_data(
                        robot_key_joints[start_idx_],
                        robot_key_joints[end_idx_])["position_vector"]
                    pos_vector /= sympy.sqrt(norm_vec[0] ** 2 + norm_vec[1] ** 2 + norm_vec[2] ** 2)

                props = {"position_vector": pos_vector}

                vector_edges.append((robot_key_joints[start_idx_], robot_key_joints[end_idx_], props))
    symbolic_graph.add_edges_from(vector_edges)
    # ====================================================================================================

    return (gc_symbols, root_rot_symbols, root_height_symbol), (gc_lower_bounds, gc_upper_bounds), \
        t_pose_graph, symbolic_graph, t_pose_toes_horizontal_distance
