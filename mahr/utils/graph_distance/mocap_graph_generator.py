import logging
import os
import pathlib
import pickle

import networkx as nx
import numpy as np
import torch

from mahr.unified_representation import UnifiedRepresentation
from mahr.utils.poselib.core.rotation3d import quat_rotate, quat_inverse

from mahr.config.MocapConfig import MocapConfig


# noinspection PyTypeChecker
def generate_vector_graphs_for_motion(
    mocap_config: MocapConfig,
    motion_dict_dir, graph_dir,
    normalize_edge_weights=False,
    force_generate_motion_dict=True,
    force_generate_graph=True,
    filter_edges_when_generating_graph=False,
    edge_filter=None
):
    logging.getLogger().info(f"Start generating graphs for mocap data {mocap_config.name}")

    # ====================================================================================================
    # Start generating motion global dict
    """
    Load previously generated global information or generate it using `UnifiedRepresentation` 
    """
    motion_dict_file = os.path.join(motion_dict_dir, f"{mocap_config.name}.pkl")
    if os.path.isfile(motion_dict_file) and (not force_generate_motion_dict):
        logging.getLogger().debug(f"Found existing file: `{motion_dict_file}`. Loading dict from file......")
        with open(motion_dict_file, "rb") as target_file:
            motion_global_dict = pickle.load(target_file)
    else:
        if force_generate_motion_dict:
            logging.getLogger().debug(f"Force generating `{motion_dict_file}`......")
        else:
            logging.getLogger().debug(f"File `{motion_dict_file}` not found, generating file......")
        motion_representation = mocap_config.uni_repr
        motion_global_dict = motion_representation.to_dict(is_local=False)
        with open(motion_dict_file, "wb") as target_file:
            pickle.dump(motion_global_dict, target_file)
    # End generating motion global dict
    # ====================================================================================================

    # ====================================================================================================
    # Start generating t-pose global dict if normalizing is required
    """
    If normalize_edge_weights==True, generate t-pose's global information for later usage
    """
    motion_t_pose_dict_file = os.path.join(motion_dict_dir, f"{mocap_config.name}_t_pose.pkl")
    if os.path.isfile(motion_t_pose_dict_file) and (not force_generate_motion_dict):
        logging.getLogger().debug(f"Found existing file: `{motion_t_pose_dict_file}`. Loading dict from file......")
        with open(motion_t_pose_dict_file, "rb") as target_file:
            motion_global_t_pose_dict = pickle.load(target_file)
    else:
        if force_generate_motion_dict:
            logging.getLogger().debug(f"Force generating `{motion_t_pose_dict_file}`......")
        else:
            logging.getLogger().debug(f"File `{motion_t_pose_dict_file}` not found, generating file......")

        # generate t pose representation from amc file
        # prior about pose of the amc description should be known in advance to give t-pose joint angles
        t_pose_representation = mocap_config.norm_pose_repr

        motion_global_t_pose_dict = t_pose_representation.to_dict(is_local=False)

        with open(motion_t_pose_dict_file, "wb") as target_file:
            pickle.dump(motion_global_t_pose_dict, target_file)

        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(1, 1, 1, projection="3d")
        # t_pose_representation.plot_frame(0, ax)
        # plt.show()

    toes_idx = [motion_global_t_pose_dict["names"].index(toe) for toe in mocap_config.toes]
    toes_translations = [motion_global_t_pose_dict["global_translations"][0][idx] for idx in toes_idx]
    t_pose_toes_horizontal_distance = np.linalg.norm(toes_translations[0][0:2] - toes_translations[1][0:2], ord=2)
    # End generating t-pose global dict
    # ====================================================================================================

    # ====================================================================================================
    # Start generating t-pose graph
    """
    generate t-pose graph for normalization
    """
    mocap_key_bodies = mocap_config.key_bodies

    num_frames = len(motion_global_dict["global_translations"])

    frame_graph_dir = os.path.join(graph_dir, mocap_config.name)
    os.makedirs(frame_graph_dir, exist_ok=True)

    # generate t-pose motion frame graph or load it from disk
    t_pose_graph_file = os.path.join(frame_graph_dir, f"t_pose.pkl")

    if os.path.isfile(t_pose_graph_file) and (not force_generate_graph):
        logging.getLogger().debug(
            f"Found existing t-pose graph for motion `{mocap_config.name}`: `{t_pose_graph_file}`. Loading......")
        with open(t_pose_graph_file, "rb") as file:
            t_pose_graph = pickle.load(file)
    else:
        # adjust t-pose root height based on toes' height
        toe_heights = []
        for toe_name in mocap_config.toes:
            toe_idx = motion_global_t_pose_dict["names"].index(toe_name)
            toe_heights.append(float(motion_global_t_pose_dict["global_translations"][0][toe_idx][2]))

        # root height of the current t-pose relative to desired on-ground height
        # equivalent to toe height of the current t-pose relative to desired on-ground height
        root_elevation = min(toe_heights)

        for body_idx in range(len(motion_global_t_pose_dict["names"])):
            motion_global_t_pose_dict["global_translations"][0][body_idx][2] -= root_elevation

        pelvis_height = motion_global_t_pose_dict["global_translations"][0][0][2]

        for body_idx in range(len(motion_global_t_pose_dict["names"])):
            motion_global_t_pose_dict["global_translations"][0][body_idx][2] += 0.05 * pelvis_height

        if force_generate_graph:
            logging.getLogger().debug(f"Force generating t-pose graph for `{mocap_config.name}`......")
        else:
            logging.getLogger().debug(f"T-pose graph for `{mocap_config.name}` not found. "
                                      f"Generating t-pose graph......")

        t_pose_graph = nx.Graph()
        t_pose_graph.add_nodes_from(mocap_key_bodies)

        vector_edges = []
        if filter_edges_when_generating_graph:
            assert edge_filter is not None
            for start_idx_, end_idx_ in edge_filter:
                if mocap_key_bodies[start_idx_] == "ground":
                    end_body_idx = motion_global_t_pose_dict["names"].index(mocap_key_bodies[end_idx_])
                    # end_body_idx = motion_global_t_pose_dict["names"].index(mocap_key_bodies[0])
                    start_body_height = 0.0
                    end_body_height = motion_global_t_pose_dict["global_translations"][0][end_body_idx][2]
                    props = {"position_vector": np.asarray([0.0, 0.0, end_body_height - start_body_height])}
                elif mocap_key_bodies[end_idx_] == "ground":
                    start_body_idx = motion_global_t_pose_dict["names"].index(mocap_key_bodies[start_idx_])
                    # start_body_idx = motion_global_t_pose_dict["names"].index(mocap_key_bodies[0])
                    start_body_height = motion_global_t_pose_dict["global_translations"][0][start_body_idx][2]
                    end_body_height = 0.0
                    props = {"position_vector": np.asarray([0.0, 0.0, end_body_height - start_body_height])}
                else:
                    start_body_idx = motion_global_t_pose_dict["names"].index(mocap_key_bodies[start_idx_])
                    end_body_idx = motion_global_t_pose_dict["names"].index(mocap_key_bodies[end_idx_])
                    start_pos_wrt_root = quat_rotate(
                        quat_inverse(
                            torch.from_numpy(motion_global_t_pose_dict["global_rotations"][0][0])
                        ),
                        torch.from_numpy(
                            motion_global_t_pose_dict["global_translations"][0][start_body_idx] -
                            motion_global_t_pose_dict["global_translations"][0][0])
                    ).numpy()
                    end_pos_wrt_root = quat_rotate(
                        quat_inverse(
                            torch.from_numpy(motion_global_t_pose_dict["global_rotations"][0][0])),
                        torch.from_numpy(
                            motion_global_t_pose_dict["global_translations"][0][end_body_idx] -
                            motion_global_t_pose_dict["global_translations"][0][0])
                    ).numpy()
                    props = {"position_vector": end_pos_wrt_root - start_pos_wrt_root}

                vector_edges.append((mocap_key_bodies[start_idx_], mocap_key_bodies[end_idx_], props))
        else:
            for start_idx_ in range(len(mocap_key_bodies)):
                for end_idx_ in range(start_idx_ + 1, len(mocap_key_bodies)):
                    if mocap_key_bodies[start_idx_] == "ground":
                        end_body_idx = motion_global_t_pose_dict["names"].index(mocap_key_bodies[end_idx_])
                        # end_body_idx = motion_global_t_pose_dict["names"].index(mocap_key_bodies[0])
                        start_body_height = 0.0
                        end_body_height = motion_global_t_pose_dict["global_translations"][0][end_body_idx][2]
                        props = {"position_vector": np.asarray([0.0, 0.0, end_body_height - start_body_height])}
                    elif mocap_key_bodies[end_idx_] == "ground":
                        start_body_idx = motion_global_t_pose_dict["names"].index(mocap_key_bodies[start_idx_])
                        # start_body_idx = motion_global_t_pose_dict["names"].index(mocap_key_bodies[0])
                        start_body_height = motion_global_t_pose_dict["global_translations"][0][start_body_idx][2]
                        end_body_height = 0.0
                        props = {"position_vector": np.asarray([0.0, 0.0, end_body_height - start_body_height])}
                    else:
                        start_body_idx = motion_global_t_pose_dict["names"].index(mocap_key_bodies[start_idx_])
                        end_body_idx = motion_global_t_pose_dict["names"].index(mocap_key_bodies[end_idx_])
                        start_pos_wrt_root = quat_rotate(
                            quat_inverse(
                                torch.from_numpy(motion_global_t_pose_dict["global_rotations"][0][0])
                            ),
                            torch.from_numpy(
                                motion_global_t_pose_dict["global_translations"][0][start_body_idx] -
                                motion_global_t_pose_dict["global_translations"][0][0])
                        ).numpy()
                        end_pos_wrt_root = quat_rotate(
                            quat_inverse(
                                torch.from_numpy(motion_global_t_pose_dict["global_rotations"][0][0])),
                            torch.from_numpy(
                                motion_global_t_pose_dict["global_translations"][0][end_body_idx] -
                                motion_global_t_pose_dict["global_translations"][0][0])
                        ).numpy()
                        props = {"position_vector": end_pos_wrt_root - start_pos_wrt_root}

                    vector_edges.append((mocap_key_bodies[start_idx_], mocap_key_bodies[end_idx_], props))
        t_pose_graph.add_edges_from(vector_edges)

        # save graph to directory
        with open(t_pose_graph_file, "wb") as target_file:
            pickle.dump(t_pose_graph, target_file)
    # End generating t-pose graph
    # ====================================================================================================

    # ====================================================================================================
    # Start generating mocap graphs
    """
    generate mocap graphs
    """
    graphs = []
    for frame in range(num_frames):
        graph_file = os.path.join(frame_graph_dir, f"{frame}.pkl")

        if os.path.isfile(graph_file) and (not force_generate_graph):
            logging.getLogger().debug(
                f"Found existing graph for motion `{mocap_config.name}` at frame `{frame}`: `{graph_file}`. Loading.")
            # save graph to directory
            with open(graph_file, "rb") as file:
                graph = pickle.load(file)
        else:
            if force_generate_graph:
                logging.getLogger().debug(f"Force generating graph for `{mocap_config.name}` at frame `{frame}`......")
            else:
                logging.getLogger().debug(f"Graph for `{mocap_config.name}` at frame `{frame}` not found. "
                                          f"Generating graph......")

            graph = nx.Graph()
            graph.add_nodes_from(mocap_key_bodies)

            vector_edges = []
            if filter_edges_when_generating_graph:
                assert edge_filter is not None
                for start_idx_, end_idx_ in edge_filter:
                    if mocap_key_bodies[start_idx_] == "ground":
                        end_body_idx = motion_global_dict["names"].index(mocap_key_bodies[end_idx_])
                        start_body_height = 0.0
                        end_body_height = motion_global_dict["global_translations"][frame][end_body_idx][2]
                        pos_vector = np.asarray([0.0, 0.0, end_body_height - start_body_height])
                    elif mocap_key_bodies[end_idx_] == "ground":
                        start_body_idx = motion_global_dict["names"].index(mocap_key_bodies[start_idx_])
                        start_body_height = motion_global_dict["global_translations"][frame][start_body_idx][2]
                        end_body_height = 0.0
                        pos_vector = np.asarray([0.0, 0.0, end_body_height - start_body_height])
                    else:
                        start_body_idx = motion_global_dict["names"].index(mocap_key_bodies[start_idx_])
                        end_body_idx = motion_global_dict["names"].index(mocap_key_bodies[end_idx_])
                        start_pos_wrt_root = quat_rotate(
                            quat_inverse(
                                torch.from_numpy(motion_global_dict["global_rotations"][frame][0])
                            ),
                            torch.from_numpy(
                                motion_global_dict["global_translations"][frame][start_body_idx] -
                                motion_global_dict["global_translations"][frame][0])
                        ).numpy()
                        end_pos_wrt_root = quat_rotate(
                            quat_inverse(
                                torch.from_numpy(motion_global_dict["global_rotations"][frame][0])),
                            torch.from_numpy(
                                motion_global_dict["global_translations"][frame][end_body_idx] -
                                motion_global_dict["global_translations"][frame][0])
                        ).numpy()

                        pos_vector = end_pos_wrt_root - start_pos_wrt_root

                    if normalize_edge_weights:
                        pos_vector /= np.linalg.norm(t_pose_graph.get_edge_data(
                            mocap_key_bodies[start_idx_], mocap_key_bodies[end_idx_]
                        )["position_vector"], ord=2) + 1e-5
                    props = {"position_vector": pos_vector}

                    vector_edges.append((mocap_key_bodies[start_idx_], mocap_key_bodies[end_idx_], props))
            else:
                for start_idx_ in range(len(mocap_key_bodies)):
                    for end_idx_ in range(start_idx_ + 1, len(mocap_key_bodies)):
                        if mocap_key_bodies[start_idx_] == "ground":
                            end_body_idx = motion_global_dict["names"].index(mocap_key_bodies[end_idx_])
                            start_body_height = 0.0
                            end_body_height = motion_global_dict["global_translations"][frame][end_body_idx][2]
                            pos_vector = np.asarray(np.asarray([0.0, 0.0, end_body_height - start_body_height]))
                        elif mocap_key_bodies[end_idx_] == "ground":
                            start_body_idx = motion_global_dict["names"].index(mocap_key_bodies[start_idx_])
                            start_body_height = motion_global_dict["global_translations"][frame][start_body_idx][2]
                            end_body_height = 0.0
                            pos_vector = np.asarray([0.0, 0.0, end_body_height - start_body_height])
                        else:
                            start_body_idx = motion_global_dict["names"].index(mocap_key_bodies[start_idx_])
                            end_body_idx = motion_global_dict["names"].index(mocap_key_bodies[end_idx_])
                            start_pos_wrt_root = quat_rotate(
                                quat_inverse(
                                    torch.from_numpy(motion_global_dict["global_rotations"][frame][0])
                                ),
                                torch.from_numpy(
                                    motion_global_dict["global_translations"][frame][start_body_idx] -
                                    motion_global_dict["global_translations"][frame][0])
                            ).numpy()
                            end_pos_wrt_root = quat_rotate(
                                quat_inverse(
                                    torch.from_numpy(motion_global_dict["global_rotations"][frame][0])),
                                torch.from_numpy(
                                    motion_global_dict["global_translations"][frame][end_body_idx] -
                                    motion_global_dict["global_translations"][frame][0])
                            ).numpy()

                            pos_vector = end_pos_wrt_root - start_pos_wrt_root
                        if normalize_edge_weights:
                            pos_vector /= np.linalg.norm(t_pose_graph.get_edge_data(
                                mocap_key_bodies[start_idx_], mocap_key_bodies[end_idx_]
                            )["position_vector"], ord=2) + 1e-5
                        props = {"position_vector": pos_vector}

                        vector_edges.append((mocap_key_bodies[start_idx_], mocap_key_bodies[end_idx_], props))
            graph.add_edges_from(vector_edges)

            # save graph to directory
            with open(graph_file, "wb") as target_file:
                pickle.dump(graph, target_file)

        graphs.append(graph)
    # End generating mocap graphs
    # ====================================================================================================

    return t_pose_graph, graphs, motion_global_dict, t_pose_toes_horizontal_distance
