# Morphology Agnostic Humanoid Retargeting via Perceptual Aligned Graph Similarity

![teaser.png](images/teaser.png)

## About this project

This project is the official implementation of the paper "Morphology-Agnostic Humanoid 
Retargeting via Perceptual Aligned Graph Similarity". The project provides a complete 
pipeline for humanoid motion retargeting. ASF/AMC and BVH formats are supported.

We provide a few retargeting examples from CMU and AMASS dataset to several robots and 
convenient extension for adding new robots.

## How to use

### Installation

You may need to create python virtual environment before executing the following commands.
Python 3.10 or higher is required.

`${ProjectDir}` in this context refers to the base directory of the project.

#### 1. Install customized `urdfpy`

[urdfpy](https://github.com/mmatl/urdfpy) is a python package for loading,
manipulating, saving, and visualizing URDF files. However, official release does not
support symbolic computation, even it only requires minor changes (
see [issue #19](https://github.com/mmatl/urdfpy/pull/19)). So I made a clone of urdfpy
code in this project under `third_party` directory. You need to install it first:

```bash
cd ${PorjectDir}/third_party/urdfpy
pip install -e .
```

#### 2. Install other requirements

```bash
cd ${ProjectDir}
pip install -r requirements.txt
```

#### 3. (Optional) Enable MuJoCo-based forward kinematics

To accelerate numeric forward-kinematics evaluations you can optionally install
[MuJoCo](https://github.com/google-deepmind/mujoco):

```bash
pip install mujoco
```

Set the environment variable before running the retargeting scripts to switch
the forward-kinematics backend from `urdfpy` to MuJoCo:

```bash
export MAHR_FK_BACKEND=mujoco
```

Alternatively, pass `fk_backend="mujoco"` when instantiating the robot config
classes (e.g. `Fourier_GR1_WholeBody_Config(fk_backend="mujoco")`). When the
variable is unset the original `urdfpy` implementation is used.

### Preparing Mocap Data

Two sources of motion capture data are supported now: CMU dataset in ASF/AMC format and 
AMASS dataset converted to BVH format.

1. **CMU dataset in ASF/AMC format**: You can acquire this format of data from their 
   official website
   [CMU Graphics Lab Motion Capture Database](https://mocap.cs.cmu.edu/). You should 
   put the data under  `mahr/rsc/all_asfamc/subjects` directory. The 
   data should follow the directory structure like this:

   ```
   mahr/rsc/all_asfamc/subjects/
   ├── 01
   │   ├── 01.asf
   │   ├── 01_01.amc
   │   └── ...
   ├── 02
   │   ├── 02.asf
   │   ├── 02_01.amc
   │   └── ...
   |── ...
   ```
   
   We provide an index file containing all the subjects in the CMU dataset under 
   directory `mahr/rsc/all_asfamc/index.json`, which will be used for 
   retargeting this dataset to robots.

2. **AMASS dataset**: Original motion data is in `npz`, which is available at 
   [AMASS](https://amass.is.tue.mpg.de/)
   You can convert the data to BVH format with 
   [CharacterAnimationTools](https://github.com/KosukeFukazawa/CharacterAnimationTools).
   You need to put the converted BVH files under directory 
   `mahr/rsc/amass_bvh`.

We provided a few examples for both formats in the above directories.

### Robot Models

We provide a few robot models in URDF format under directory 
`mahr/rsc/assets/urdf`. We modified the original URDF files to
make them compatible with our retargeting algorithm. Basically, we added some "marker" 
links and joints to the URDF files to serve as critical landmarks for retargeting. For 
example, in `Berkeley_Humanoid_Lite/berkeley_humanoid_lite_stl.urdf`, the marker joint 
and link for pelvis look like this:

```xml
<joint name="pelvis_marker_joint" type="fixed">
  <origin xyz="0 0 0.60" rpy="0 0 0"/>
  <parent link="base"/>
  <child link="pelvis_marker"/>
</joint>
<link name="pelvis_marker">
<visual>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <sphere radius="0.08"/>
  </geometry>
  <material name="dark"/>
</visual>
</link>
```

Note that we did not add additional DoFs to the robot with the added fixed joint.

### Retargeting Motion Capture Data to Robots

In the `motion_retarget_graph_distance.py` script, we provided several examples for 
retargeting mocap data to robots. There examples are in the form of functions starting 
with `main_`, such as `main_CMU_to_GR1`. Uncomment the function you want to run at the 
end of the script, and run the script. The retargeted motion will be saved in 
directories similar to `graph_distance_xxx/optimized_trajectories_vector_graph/` in 
json format, which contains the retargeted joint angles and root states.

Some example retargeted motion JSON files are provided under the `retargeted_motions/`
directory.

### Visualizing Retargeted Motions

Use `visualize_retargeted_motion.py` to play back a retargeted trajectory on the
corresponding robot URDF. For example, to visualize a Unitree G1 trajectory from the
provided examples:

```bash
python visualize_retargeted_motion.py \
  -t retargeted_motions/UnitreeG1/mahr/80_43_poses.json \
  -u mahr/rsc/assets/urdf/unitree_g1/g1_29dof_stl.urdf \
  --no-loop
```

The `-t/--trajectory` argument points to the retargeted JSON file, and
`-u/--urdf` points to the robot URDF used for playback.

Use the following URDF paths for the supported robots:

| Robot | URDF path |
| --- | --- |
| Agibot X1 | `mahr/rsc/assets/urdf/Agibot_X1/urdf/x1_stl.urdf` |
| Berkeley Humanoid Lite | `mahr/rsc/assets/urdf/Berkeley_Humanoid_Lite/berkeley_humanoid_lite_stl.urdf` |
| Boston Dynamics Atlas HD | `mahr/rsc/assets/urdf/Boston_Dynamics_Atlas/urdf/atlas_v4_with_multisense_stl.urdf` |
| Fourier GR1 | `mahr/rsc/assets/urdf/Fourier_GR1/GR1.urdf` |
| Fourier N1 | `mahr/rsc/assets/urdf/Fourier_N1/urdf/N1_raw_stl.urdf` |
| iCub | `mahr/rsc/assets/urdf/iCub/robots/iCubGazeboV2_7/model_stl.urdf` |
| Unitree G1 | `mahr/rsc/assets/urdf/unitree_g1/g1_29dof_stl.urdf` |
| Unitree H1 | `mahr/rsc/assets/urdf/unitree_h1/urdf/h1_stl.urdf` |

### Acknowledgements

We would like to thank the following projects for their contributions:

- [urdfpy](https://github.com/mmatl/urdfpy) for loading URDF files and calculating
  forward kinematics.
- [MuJoCo](https://github.com/google-deepmind/mujoco) for fast numeric forward
  kinematics.
- [AMCParser](https://github.com/CalciferZh/AMCParser) for parsing CMU Mocap's ASF/AMC
  files.
- [ASE](https://github.com/nv-tlabs/ASE), which we use its poselib component for 
  intermediate pose representation.
