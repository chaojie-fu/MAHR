import os

urdf_rsc_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../rsc/assets/urdf")
)

fourier_gr1_urdf = os.path.join(urdf_rsc_dir, "Fourier_GR1/GR1.urdf")

unitree_g1_urdf = os.path.join(urdf_rsc_dir, "unitree_g1/g1_29dof_stl.urdf")

unitree_h1_urdf = os.path.join(urdf_rsc_dir, "unitree_h1/urdf/h1_stl.urdf")

berkeley_humanoid_lite_urdf = os.path.join(urdf_rsc_dir, "Berkeley_Humanoid_Lite/berkeley_humanoid_lite_stl.urdf")

fourier_n1_urdf = os.path.join(urdf_rsc_dir, "Fourier_N1/urdf/N1_raw_stl.urdf")

boston_dynamics_atlas_urdf = os.path.join(urdf_rsc_dir, "Boston_Dynamics_Atlas/urdf/atlas_v4_with_multisense_stl.urdf")

agibot_x1_urdf = os.path.join(urdf_rsc_dir, "Agibot_X1/urdf/x1_stl.urdf")

icub_urdf = os.path.join(urdf_rsc_dir, "iCub/robots/iCubGazeboV2_7/model_stl.urdf")
