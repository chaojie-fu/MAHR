# python
import sys
from line_profiler import LineProfiler
from mocap_data_converter.utils.mujoco_fk import MujocoURDFRobot
import numpy as np
import mujoco

URDF_PATH = "mocap_data_converter/rsc/assets/urdf/Fourier_GR1/GR1.urdf"  # <- set your URDF path
ITERATIONS = 200  # increase for more stable measurements

robot = MujocoURDFRobot(URDF_PATH)

# Build a representative joint config (dict by name)
cfg = {name: 0.1 for name in robot.actuated_joint_names}

def workload():
    # call link_fk repeated times (profiling benefits from multiple calls)
    for _ in range(ITERATIONS):
        # use_names=True to exercise the mujo co fast path in your code
        _ = robot.link_fk(cfg=cfg, use_names=True)


# mujoco.mj_saveLastXML("robot.xml", robot._model)

lp = LineProfiler()
lp.add_function(MujocoURDFRobot.link_fk)
lp_wrapper = lp(workload)  # wrap the workload
lp_wrapper()               # run workload under the profiler

# Print human-readable per-line stats
lp.print_stats(stream=sys.stdout)

# workload()
