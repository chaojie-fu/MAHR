# ====================IMPORTANT INFORMATION0====================
# ALL quat of class members use format [x, y, z, w]
# ====================IMPORTANT INFORMATION0====================

from typing import List, Dict, Union

import bvhio
import mpl_toolkits
import numpy as np
import torch
from bvhio.lib.bvh import BvhJoint

from mahr.utils.poselib.core.rotation3d import quat_from_rotation_matrix, \
    quat_mul_norm, quat_rotate
from .utils.asfamcparser.asf_amc_parser import parse_asf, parse_amc, Joint
from .utils.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree


class LocalJoint:
    def __init__(self, joint_name, local_translation, local_rotation, parent=None):
        # local translation and local rotation defined same as SkeletonState
        self._name = joint_name
        self._local_translation = local_translation
        self._local_rotation = local_rotation
        self.parent = parent
        if parent is not None:
            parent.children.append(self)
        self.children = []

    def local_translation(self, z_up=False):
        if z_up:
            # shape: [3]
            return self._local_translation[[2, 0, 1], 0]
        else:
            # shape: [3]
            return self._local_translation[[2, 0, 1], 0]

    def local_rotation(self, z_up=False, quat=False):
        if z_up:
            if quat:
                # shape: [4]
                return quat_from_rotation_matrix(torch.from_numpy(
                    self._local_rotation[[2, 0, 1], :][:, [2, 0, 1]])).numpy()
            else:
                # shape: [3, 3]
                return self._local_rotation[[2, 0, 1], :][:, [2, 0, 1]]
        else:
            if quat:
                # shape: [4]
                return quat_from_rotation_matrix(torch.from_numpy(
                    self._local_rotation)).numpy()
            else:
                # shape: [3, 3]
                return self._local_rotation

    @property
    def name(self):
        return self._name

    def to_list(self, z_up=True):
        assert self.parent is None, "ERROR: This method works only for root node!"
        translation_output = []
        rotation_output = []
        name_output = []
        parent_indices_output = []

        def add_node(node: LocalJoint, name_list, parent_indices_list, local_translation_list, local_rotation_list,
                     is_z_up=True):
            if node.parent is None:
                parent_indices_list.append(-1)
            else:
                parent_indices_list.append(name_list.index(node.parent.name))

            name_list.append(node.name)
            local_translation_list.append(np.expand_dims(node.local_translation(z_up=is_z_up), axis=0))
            local_rotation_list.append(np.expand_dims(node.local_rotation(z_up=is_z_up, quat=True), axis=0))
            if len(node.children) == 0:
                return
            else:
                for child in node.children:
                    add_node(child, name_list, parent_indices_list, local_translation_list, local_rotation_list,
                             is_z_up=is_z_up)

        add_node(self, name_output, parent_indices_output, translation_output, rotation_output, is_z_up=z_up)

        return name_output, parent_indices_output, \
            np.concatenate(translation_output, axis=0), \
            np.concatenate(rotation_output, axis=0)


class UnifiedRepresentation:
    """
    Unified Representation for converting among different formats
    Now supports:
        From:
          - asf/amc from CMU database (may work for other asf/amc files from other database, but not tested)
          - bvh
          - python native dict with self-defined structure
          - SkeletonState from ASE's (https://github.com/nv-tlabs/ASE/tree/main) poselib
        To:
          - python native dict with self-defined structure
          - SkeletonState & SkeletonMotion from ASE's (https://github.com/nv-tlabs/ASE/tree/main) poselib
    """

    def __init__(self):
        self._num_joints = 0
        # list of joint names
        self._names = []
        # list of integers, which indicates the index of the parent joint
        self._parent_indices = []
        # translations are of size [num_frames], whose elements are numpy arrays of shape [num_joints, 3]
        self._local_translations = []
        # reference_translations are of size [num_frames], whose elements are numpy arrays of shape [num_joints, 3]
        self._reference_rotations = []
        # rotations are of size [num_frames], whose elements are numpy arrays of shape [num_joints, 4],
        # with quaternion format [x, y, z, w]
        # local rotation is the rotation of local frame relative to parent frame
        self._local_rotations = []
        self._global_translations = None
        self._global_rotations = None
        self._fps = 0

    def _reset(self):
        self._num_joints = 0
        # list of joint names
        self._names = []
        # list of integers, which indicates the index of the parent joint
        self._parent_indices = []
        # translations are of size [num_frames], whose elements are numpy arrays of shape [num_joints, 3]
        self._local_translations = []
        # reference_translations are of size [num_frames], whose elements are numpy arrays of shape [num_joints, 3]
        self._reference_rotations = []
        # rotations are of size [num_frames], whose elements are numpy arrays of shape [num_joints, 4],
        # with quaternion format [x, y, z, w]
        # local rotation is the rotation of local frame relative to parent frame
        self._local_rotations = []
        self._global_translations = None
        self._global_rotations = None
        self._fps = 0

    def to_dict(self, is_local=True):
        if is_local:
            return {
                "names": self._names.copy(),
                "parent_indices": self._parent_indices.copy(),
                "local_translations": [item.copy() for item in self._local_translations],
                "reference_rotations": [item.copy() for item in self._reference_rotations],
                "local_rotations": [item.copy() for item in self._local_rotations],
                "fps": self._fps
            }
        else:
            self._local_to_global()
            return {
                "names": self._names.copy(),
                "parent_indices": self._parent_indices.copy(),
                "global_translations": [item.copy() for item in self._global_translations],
                "reference_rotations": [item.copy() for item in self._reference_rotations],
                "global_rotations": [item.copy() for item in self._global_rotations],
                "fps": self._fps
            }

    def from_dict(self, dict_repr: dict):
        self._reset()

        assert "names" in dict_repr.keys()
        assert "parent_indices" in dict_repr.keys()
        assert "local_translations" in dict_repr.keys()
        assert "reference_rotations" in dict_repr.keys()
        assert "local_rotations" in dict_repr.keys()
        assert "fps" in dict_repr.keys()

        self._names = dict_repr["names"]
        self._parent_indices = dict_repr["parent_indices"]
        self._local_translations = dict_repr["local_translations"]
        self._reference_rotations = dict_repr["reference_rotations"]
        self._local_rotations = dict_repr["local_rotations"]
        self._fps = dict_repr["fps"]
        self._num_joints = len(self._names)

        return self

    def from_asf_amc(self, asf_file: str, amc_file: str, fps=0):
        self._reset()

        self._fps = fps
        joints = parse_asf(asf_file)
        frames = parse_amc(amc_file)

        # ====================================================================================================
        # Convert tree of Joint to name and index list
        root_node = joints['root']

        for idx, frame in enumerate(frames):
            root_node.set_motion(frame)
            local_joints = dict()

            self._add_children(local_joints, root_node, None)

            names, parent_indices, local_translations, local_rotations = local_joints["root_auxiliary"].to_list()
            if idx == 0:
                self._names = names
                self._parent_indices = parent_indices
                self._num_joints = len(self._names)
            self._local_translations.append(local_translations)
            reference_rotations = np.zeros_like(local_rotations, dtype=np.float32)
            reference_rotations[:, 3] = 1.0
            self._reference_rotations.append(reference_rotations)
            self._local_rotations.append(local_rotations)
        self._local_to_global()

    def t_pose_from_asf(self, asf_file, t_pose_joint_angles_in_degree: Dict):
        self._reset()
        joints = parse_asf(asf_file)

        # ====================================================================================================
        # Convert tree of Joint to name and index list
        root_node = joints['root']

        root_node.set_motion(t_pose_joint_angles_in_degree)
        local_joints = dict()

        self._add_children(local_joints, root_node, None)

        names, parent_indices, local_translations, local_rotations = local_joints["root_auxiliary"].to_list()

        self._names = names
        self._parent_indices = parent_indices
        self._num_joints = len(self._names)
        self._local_translations.append(local_translations)
        reference_rotations = np.zeros_like(local_rotations, dtype=np.float32)
        reference_rotations[:, 3] = 1.0
        self._reference_rotations.append(reference_rotations)
        self._local_rotations.append(local_rotations)
        self._local_to_global()

    def from_bvh(self, bvh_file, z_up=True, motion_range=None):
        """

        :param bvh_file: path of .bvh file.
        :param z_up: if the data is converted to z_up form (only y_up bvh files are supported)
        :param motion_range: loaded motion range
        :return: None
        """
        self._reset()

        bvh = bvhio.readAsBvh(bvh_file)
        self._fps = 1.0 / bvh.FrameTime

        bvh_joints: List[BvhJoint] = []

        # extract joints from BvhContainer (convert from tree to lists)
        def add_children(bvh_joint: BvhJoint, parent_name):
            self._names.append(bvh_joint.Name)
            bvh_joints.append(bvh_joint)
            if parent_name is None:
                self._parent_indices.append(-1)
            else:
                self._parent_indices.append(self._names.index(parent_name))

            if len(bvh_joint.Children) > 0:
                for child in bvh_joint.Children:
                    add_children(child, bvh_joint.Name)

        add_children(bvh.Root, None)

        self._num_joints = len(bvh_joints)

        if motion_range is None:
            frames = range(bvh.FrameCount)
        else:
            frames = range(motion_range[0], motion_range[1])
        for frame in frames:
            local_translation = np.zeros([self._num_joints, 3])
            local_rotation = np.zeros([self._num_joints, 4])
            reference_rotation = np.zeros([self._num_joints, 4])
            reference_rotation[:, 3] = 1.0

            for joint in range(self._num_joints):
                if z_up:
                    local_translation[joint, 0] = bvh_joints[joint].Keyframes[frame].Position.z
                    local_translation[joint, 1] = bvh_joints[joint].Keyframes[frame].Position.x
                    local_translation[joint, 2] = bvh_joints[joint].Keyframes[frame].Position.y
                else:
                    local_translation[joint, 0] = bvh_joints[joint].Keyframes[frame].Position.x
                    local_translation[joint, 1] = bvh_joints[joint].Keyframes[frame].Position.y
                    local_translation[joint, 2] = bvh_joints[joint].Keyframes[frame].Position.z

                if z_up:
                    local_rotation[joint, 0] = bvh_joints[joint].Keyframes[frame].Rotation.z
                    local_rotation[joint, 1] = bvh_joints[joint].Keyframes[frame].Rotation.x
                    local_rotation[joint, 2] = bvh_joints[joint].Keyframes[frame].Rotation.y
                else:
                    local_rotation[joint, 0] = bvh_joints[joint].Keyframes[frame].Rotation.x
                    local_rotation[joint, 1] = bvh_joints[joint].Keyframes[frame].Rotation.y
                    local_rotation[joint, 2] = bvh_joints[joint].Keyframes[frame].Rotation.z
                local_rotation[joint, 3] = bvh_joints[joint].Keyframes[frame].Rotation.w

            self._local_translations.append(local_translation)
            self._reference_rotations.append(reference_rotation)
            self._local_rotations.append(local_rotation)

        self._local_to_global()

    def from_skeleton_state(self, skeleton_state: SkeletonState, fps=0):
        self._reset()

        self._names = skeleton_state.skeleton_tree.node_names
        self._num_joints = len(self._names)
        self._parent_indices = skeleton_state.skeleton_tree.parent_indices.tolist()
        self._fps = fps
        if len(skeleton_state.tensor.shape) == 1:
            self._local_translations = [skeleton_state.local_translation.numpy()]
            self._local_rotations = [skeleton_state.local_rotation.numpy()]
            self._reference_rotations = [skeleton_state.reference_rotation.numpy()]
        else:
            self._local_translations = [skeleton_state.local_translation[idx, ...].numpy() for idx in
                                        range(len(skeleton_state))]
            self._local_rotations = [skeleton_state.local_rotation[idx, ...].numpy() for idx in
                                     range(len(skeleton_state))]
            self._reference_rotations = [skeleton_state.reference_rotation[idx, ...].numpy() for idx in
                                         range(len(skeleton_state))]
        self._local_to_global()

    def to_skeleton_state(self, is_local=True):
        names = self._names
        parent_indices = torch.tensor(self._parent_indices, dtype=torch.long)

        if is_local:
            local_translation = torch.tensor(self._local_translations[0], dtype=torch.float32)
            reference_rotation = torch.tensor(self._reference_rotations[0], dtype=torch.float32)
            skeleton_tree = SkeletonTree(names, parent_indices, local_translation, reference_rotation)

            flattened_local_rotation = torch.cat(
                [torch.tensor(numpy_array).reshape(1, -1) for numpy_array in self._local_rotations], dim=0)
            root_index = self._parent_indices.index(-1)
            root_translation = torch.cat(
                [torch.tensor(numpy_array)[[root_index], :] for numpy_array in self._local_translations], dim=0)

            tensor_backend = torch.cat((flattened_local_rotation, root_translation), dim=-1)

        else:
            self._local_to_global()
            local_translation = torch.tensor(self._local_translations[0], dtype=torch.float32)
            reference_rotation = torch.tensor(self._reference_rotations[0], dtype=torch.float32)
            skeleton_tree = SkeletonTree(names, parent_indices, local_translation, reference_rotation)

            flattened_global_rotation = torch.cat(
                [torch.tensor(numpy_array).reshape(1, -1) for numpy_array in self._global_rotations], dim=0)
            root_index = self._parent_indices.index(-1)
            root_translation = torch.cat(
                [torch.tensor(numpy_array)[[root_index], :] for numpy_array in self._global_translations], dim=0)

            tensor_backend = torch.cat((flattened_global_rotation, root_translation), dim=-1)

        skeleton_state = SkeletonState(tensor_backend, skeleton_tree, is_local)

        return skeleton_state

    def to_skeleton_motion(self, is_local=True):
        skeleton_state = self.to_skeleton_state(is_local=is_local)
        return SkeletonMotion.from_skeleton_state(skeleton_state, fps=self.fps)

    @staticmethod
    def _add_children(target_tree_node_dict: Dict,
                      curr_node_in_source_tree: Joint,
                      parent_node_in_target_tree: Union[LocalJoint, None]):
        curr_node_name = curr_node_in_source_tree.name
        if curr_node_name == "root":
            aux_node_name = f"{curr_node_name}_auxiliary"
            root_aux_node = LocalJoint(aux_node_name,
                                       curr_node_in_source_tree.coordinate,
                                       curr_node_in_source_tree.local_rotation)
            target_tree_node_dict.update({aux_node_name: root_aux_node})

            children: List[Joint] = curr_node_in_source_tree.children

            if len(children) > 1:
                node_root = LocalJoint(curr_node_name,
                                       curr_node_in_source_tree.local_translation,
                                       np.eye(3, dtype=np.float32),
                                       parent=root_aux_node)
                target_tree_node_dict.update({curr_node_name: node_root})

                for child in children:
                    child_aux_name = f"{child.name}_auxiliary"
                    node = LocalJoint(child_aux_name,
                                      np.zeros((3, 1), dtype=np.float32),
                                      child.local_rotation,
                                      parent=node_root)
                    target_tree_node_dict.update({child_aux_name: node})
                    UnifiedRepresentation._add_children(target_tree_node_dict, child, node)
            elif len(children) == 1:
                child = children[0]
                node_root = LocalJoint(curr_node_name,
                                       curr_node_in_source_tree.local_translation,
                                       child.local_rotation,
                                       parent=root_aux_node)
                target_tree_node_dict.update({curr_node_name: node_root})
                UnifiedRepresentation._add_children(target_tree_node_dict, child, node_root)
            else:
                return

        else:
            curr_node_name = curr_node_in_source_tree.name
            children = curr_node_in_source_tree.children
            if len(children) > 1:
                node = LocalJoint(curr_node_name,
                                  curr_node_in_source_tree.local_translation,
                                  np.eye(3, dtype=np.float32),
                                  parent=parent_node_in_target_tree)
                target_tree_node_dict.update({curr_node_name: node})

                for child in children:
                    child_name = child.name
                    aux_name = f"{child_name}_auxiliary"
                    child_aux_node = LocalJoint(aux_name,
                                                np.zeros((3, 1), dtype=np.float32),
                                                child.local_rotation,
                                                parent=node)
                    target_tree_node_dict.update({aux_name: child_aux_node})

                    UnifiedRepresentation._add_children(target_tree_node_dict, child, child_aux_node)
            elif len(children) == 1:
                child = children[0]
                node = LocalJoint(curr_node_name,
                                  curr_node_in_source_tree.local_translation,
                                  child.local_rotation,
                                  parent=parent_node_in_target_tree)

                target_tree_node_dict.update({curr_node_name: node})

                UnifiedRepresentation._add_children(target_tree_node_dict, child, node)
            else:
                node = LocalJoint(curr_node_name,
                                  curr_node_in_source_tree.local_translation,
                                  np.eye(3, dtype=np.float32),
                                  parent=parent_node_in_target_tree)
                target_tree_node_dict.update({curr_node_name: node})
                return

    def _local_to_global(self):
        if (self._global_translations is None) or (self._global_rotations is None):
            num_frames = len(self)
            num_joints = self._num_joints

            self._global_translations = [np.zeros([num_joints, 3], dtype=np.float32) for _ in range(num_frames)]
            self._global_rotations = [np.zeros([num_joints, 4], dtype=np.float32) for _ in range(num_frames)]
            for frame in range(num_frames):
                frame_translation = np.zeros([num_joints, 3], dtype=np.float32)
                frame_rotation = np.zeros([num_joints, 4], dtype=np.float32)
                for idx in range(num_joints):
                    parent_idx = self._parent_indices[idx]
                    if parent_idx == -1:
                        frame_translation[idx, :] = self._local_translations[frame][idx, :].copy()
                        frame_rotation[idx, :] = quat_mul_norm(
                            torch.from_numpy(self._reference_rotations[frame][idx, :]),
                            torch.from_numpy(self._local_rotations[frame][idx, :])).numpy()
                    else:
                        # quaternions in [x, y, z, w] format
                        local_rot = self._local_rotations[frame][idx, :]
                        local_vec = self._local_translations[frame][idx, :]
                        global_parent_rot = frame_rotation[parent_idx, :]
                        global_parent_vec = frame_translation[parent_idx]
                        reference_rot = self._reference_rotations[frame][idx, :]

                        frame_rotation[idx, :] = quat_mul_norm(
                            torch.from_numpy(global_parent_rot),
                            quat_mul_norm(
                                torch.from_numpy(reference_rot),
                                torch.from_numpy(local_rot)
                            )).numpy()
                        frame_translation[idx, :] = global_parent_vec + quat_rotate(
                            torch.from_numpy(global_parent_rot),
                            torch.from_numpy(local_vec)).numpy()

                self._global_translations[frame] = frame_translation
                self._global_rotations[frame] = frame_rotation

    @property
    def num_joints(self):
        return len(self._names)

    @property
    def num_frames(self):
        return len(self)

    @property
    def fps(self):
        return self._fps

    def plot_frame(self, frame, ax: mpl_toolkits.mplot3d.Axes3D, label_joints=False):
        self._local_to_global()
        x_bounds = [0.0, 0.0]
        y_bounds = [0.0, 0.0]
        z_bounds = [0.0, 0.0]

        root_name = self._names[0]
        root_pos = self._global_translations[frame][0, :]
        ax.scatter(root_pos[0], root_pos[1], root_pos[2], c="blue")
        if label_joints:
            ax.text(root_pos[0] + 0.1, root_pos[1] + 0.1, root_pos[2],
                    root_name,
                    fontsize=10,
                    color="black",
                    ha="left",
                    va="bottom")

        for joint_idx in range(1, len(self._parent_indices)):
            parent_idx = self._parent_indices[joint_idx]
            joint_name = self._names[joint_idx]

            joint_pos = self._global_translations[frame][joint_idx, :]
            parent_pos = self._global_translations[frame][parent_idx, :]

            if joint_idx == 1:
                x_bounds[0] = joint_pos[0]
                x_bounds[1] = joint_pos[0]

                y_bounds[0] = joint_pos[1]
                y_bounds[1] = joint_pos[1]

                z_bounds[0] = joint_pos[2]
                z_bounds[1] = joint_pos[2]
            else:
                x_bounds[0] = min(joint_pos[0], x_bounds[0])
                x_bounds[1] = max(joint_pos[0], x_bounds[1])

                y_bounds[0] = min(joint_pos[1], y_bounds[0])
                y_bounds[1] = max(joint_pos[1], y_bounds[1])

                z_bounds[0] = min(joint_pos[2], z_bounds[0])
                z_bounds[1] = max(joint_pos[2], z_bounds[1])

            ax.plot([parent_pos[0], joint_pos[0]],
                    [parent_pos[1], joint_pos[1]],
                    [parent_pos[2], joint_pos[2]], 'b-')
            ax.scatter(joint_pos[0], joint_pos[1], joint_pos[2], c="blue")
            if label_joints:
                ax.text(joint_pos[0] + 0.1, joint_pos[1] + 0.1, joint_pos[2],
                        joint_name,
                        fontsize=10,
                        color="black",
                        ha="left",
                        va="bottom")

        max_interval = max(x_bounds[1] - x_bounds[0],
                           y_bounds[1] - y_bounds[0],
                           z_bounds[1] - z_bounds[0])
        axis_x_bounds = [(x_bounds[0] + x_bounds[1]) / 2.0 - max_interval / 2.0,
                         (x_bounds[0] + x_bounds[1]) / 2.0 + max_interval / 2.0]
        axis_y_bounds = [(y_bounds[0] + y_bounds[1]) / 2.0 - max_interval / 2.0,
                         (y_bounds[0] + y_bounds[1]) / 2.0 + max_interval / 2.0]
        axis_z_bounds = [(z_bounds[0] + z_bounds[1]) / 2.0 - max_interval / 2.0,
                         (z_bounds[0] + z_bounds[1]) / 2.0 + max_interval / 2.0]

        ax.set_xlim(xmin=axis_x_bounds[0], xmax=axis_x_bounds[1])
        ax.set_ylim(ymin=axis_y_bounds[0], ymax=axis_y_bounds[1])
        ax.set_zlim(zmin=axis_z_bounds[0], zmax=axis_z_bounds[1])

    def get_root_global_pos(self, frame):
        self._local_to_global()
        return self._global_translations[frame][0]

    def __len__(self):
        """ number of frames
        :return: number of frames
        """
        return len(self._local_translations)
