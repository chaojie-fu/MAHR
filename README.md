# Morphology_Agnostic_Humanoid_Retargeting_via_Perceptual_Aligned_Graph_Similarity

## About this project

This project is the official implementation of the paper "Morphology-Agnostic Humanoid 
Retargeting via Perceptual Aligned Graph Similarity". The project provides a complete 
pipeline for humanoid motion retargeting. Asf/amc and BVH formats are supported.

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

### Preparing Mocap Data

Two sources of motion capture data are supported now: CMU dataset in asf/amc format and 
AMASS dataset converted to BVH format.

1. **CMU dataset in asf/amc format**: You can acquire this format of data from their 
   official website
   [CMU Graphics Lab Motion Capture Database](https://mocap.cs.cmu.edu/). You should 
   put the data under  `mocap_data_converter/rsc/all_asfamc/subjects` directory. The 
   data should follow the directory structure like this:

   ```
   mocap_data_converter/rsc/all_asfamc/subjects/
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
   directory `mocap_data_converter/rsc/all_asfamc/index.json`, which will be used for 
   retargeting this dataset to robots.

2. **AMASS dataset**: Original motion data is in `npz`, which is available at 
   [AMASS](https://amass.is.tue.mpg.de/)
   You can convert the data to BVH format with 
   [CharacterAnimationTools](https://github.com/KosukeFukazawa/CharacterAnimationTools).
   You need to put the converted BVH files under directory 
   `mocap_data_converter/rsc/amass_bvh`.

We provided a few examples for both formats in the above directories.

### Robot Models

We provide a few robot models in URDF format under directory 
`mocap_data_converter/rsc/assets/urdf`. We modified the original URDF files to
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

### Acknowledgements

TODO