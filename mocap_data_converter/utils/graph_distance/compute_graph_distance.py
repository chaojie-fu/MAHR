import logging

import numpy as np

OPTIM_SOFT_MAX_COEFF = 5.0


def compute_vector_graph_distance(
    mocap_graph, mocap_key_bodies,
    robot_graph, robot_key_joints,
    flag_weighted_optimization_by_closeness=False,
    flag_filter_edges_when_generating_graph=False,
    edge_filter=None
):
    logging.getLogger().info("Start computing symbolic graphs distance expression")
    mocap_edge_pos_vecs = []
    robot_edge_pos_vecs = []
    if flag_filter_edges_when_generating_graph:
        assert edge_filter is not None
        for start_idx, end_idx in edge_filter:
            mocap_start_body = mocap_key_bodies[start_idx]
            mocap_end_body = mocap_key_bodies[end_idx]
            robot_start_body = robot_key_joints[start_idx]
            robot_end_body = robot_key_joints[end_idx]

            mocap_edge_pos_vecs.append(mocap_graph.get_edge_data(mocap_start_body, mocap_end_body)["position_vector"])
            robot_edge_pos_vecs.append(robot_graph.get_edge_data(robot_start_body, robot_end_body)["position_vector"])

    else:
        for start_idx in range(len(mocap_key_bodies)):
            for end_idx in range(start_idx + 1, len(mocap_key_bodies)):
                mocap_start_body = mocap_key_bodies[start_idx]
                mocap_end_body = mocap_key_bodies[end_idx]
                robot_start_body = robot_key_joints[start_idx]
                robot_end_body = robot_key_joints[end_idx]

                mocap_edge_pos_vecs.append(
                    mocap_graph.get_edge_data(mocap_start_body, mocap_end_body)["position_vector"])
                robot_edge_pos_vecs.append(
                    robot_graph.get_edge_data(robot_start_body, robot_end_body)["position_vector"])

    mocap_edge_pos_vecs_np = np.asarray(mocap_edge_pos_vecs)
    robot_edge_pos_vecs_np = np.asarray(robot_edge_pos_vecs)

    if flag_weighted_optimization_by_closeness:
        mocap_edge_distance_exp = np.exp(
            -1.0 * OPTIM_SOFT_MAX_COEFF * np.linalg.norm(mocap_edge_pos_vecs_np, ord=2, axis=-1))
        mocap_optim_weight_dominator = np.sum(mocap_edge_distance_exp)
        mocap_optim_weight = mocap_edge_distance_exp / mocap_optim_weight_dominator

        distance = np.sum(
            np.dot(mocap_optim_weight,
                   np.square(mocap_edge_pos_vecs_np - robot_edge_pos_vecs_np)))
    else:
        distance = np.sum(np.square(mocap_edge_pos_vecs_np - robot_edge_pos_vecs_np))

    return distance


# mocap_graph and robot_graph should be NORMALIZED graphs
# the un-normalized version is not taken into consideration yet
def compute_vector_graph_distance_v2(
    mocap_graph, mocap_key_bodies, mocap_t_pose_graph,
    robot_graph, robot_key_joints, robot_t_pose_graph,
    flag_weighted_optimization_by_closeness=False,
    flag_filter_edges_when_generating_graph=False,
    edge_filter=None
):
    logging.getLogger().info("Start computing symbolic graphs distance expression")
    mocap_edge_pos_vecs = []
    mocap_norm_edge_pos_vecs = []
    robot_edge_pos_vecs = []
    robot_norm_edge_pos_vecs = []
    if flag_filter_edges_when_generating_graph:
        assert edge_filter is not None
        for start_idx, end_idx in edge_filter:
            mocap_start_body = mocap_key_bodies[start_idx]
            mocap_end_body = mocap_key_bodies[end_idx]
            robot_start_body = robot_key_joints[start_idx]
            robot_end_body = robot_key_joints[end_idx]

            mocap_edge_pos_vecs.append(
                mocap_graph.get_edge_data(mocap_start_body, mocap_end_body)["position_vector"])
            robot_edge_pos_vecs.append(
                robot_graph.get_edge_data(robot_start_body, robot_end_body)["position_vector"])
            mocap_norm_edge_pos_vecs.append(
                mocap_t_pose_graph.get_edge_data(mocap_start_body, mocap_end_body)["position_vector"])
            robot_norm_edge_pos_vecs.append(
                robot_t_pose_graph.get_edge_data(robot_start_body, robot_end_body)["position_vector"])

    else:
        for start_idx in range(len(mocap_key_bodies)):
            for end_idx in range(start_idx + 1, len(mocap_key_bodies)):
                mocap_start_body = mocap_key_bodies[start_idx]
                mocap_end_body = mocap_key_bodies[end_idx]
                robot_start_body = robot_key_joints[start_idx]
                robot_end_body = robot_key_joints[end_idx]

                mocap_edge_pos_vecs.append(
                    mocap_graph.get_edge_data(mocap_start_body, mocap_end_body)["position_vector"])
                robot_edge_pos_vecs.append(
                    robot_graph.get_edge_data(robot_start_body, robot_end_body)["position_vector"])
                mocap_norm_edge_pos_vecs.append(
                    mocap_t_pose_graph.get_edge_data(mocap_start_body, mocap_end_body)["position_vector"])
                robot_norm_edge_pos_vecs.append(
                    robot_t_pose_graph.get_edge_data(robot_start_body, robot_end_body)["position_vector"])

    mocap_edge_pos_vecs_np = np.asarray(mocap_edge_pos_vecs)
    robot_edge_pos_vecs_np = np.asarray(robot_edge_pos_vecs)
    mocap_norm_edge_pos_vecs_np = np.asarray(mocap_norm_edge_pos_vecs)
    robot_norm_edge_pos_vecs_np = np.asarray(robot_norm_edge_pos_vecs)

    if flag_weighted_optimization_by_closeness:
        mocap_norm_edge_length = np.linalg.norm(mocap_norm_edge_pos_vecs_np, axis=1)
        mocap_origin_vec = mocap_edge_pos_vecs_np * np.tile(mocap_norm_edge_length.reshape([-1, 1]), (1, 3))
        mocap_edge_length = np.linalg.norm(mocap_origin_vec, axis=1)
        mocap_optim_weight = np.square(np.divide(mocap_norm_edge_length, mocap_edge_length))

        distance = np.sum(
            np.dot(mocap_optim_weight,
                   np.square(mocap_edge_pos_vecs_np - robot_edge_pos_vecs_np)))
    else:
        distance = np.sum(np.square(mocap_edge_pos_vecs_np - robot_edge_pos_vecs_np))

    return distance
