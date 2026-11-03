# MAHR v2

面向单条动作延迟重新设计的重定向核心。原 `../MAHR` 保持不变；新版不导入旧版运行时，默认只读取旧仓库提供的 URDF、动作和归一化姿态资产。

方法来源：[Fu et al., 2026, Morphology-agnostic humanoid retargeting via perception-motivated graph similarity](https://doi.org/10.3389/fnbot.2026.1814977)。**论文与代码发生差异时，以旧代码为准。** 详细决策见 [DESIGN.md](DESIGN.md)。

## 运行

Python 3.10+：

```bash
cd /home/cjfu/Vibe_Projects/MAHR_v2
python -m pip install -e '.[test]'
mahr-v2 ../MAHR/mahr/rsc/amass_bvh/DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh \
  --config configs/g1_stand.json -o results/dance.json
```

无需安装到环境也可以运行。当前机器已有的 `gmr` 环境满足新版运行依赖：

```bash
PYTHONPATH=src /home/cjfu/miniconda3/envs/gmr/bin/python -m mahr_v2 \
  ../MAHR/mahr/rsc/amass_bvh/DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh \
  --config configs/g1_stand.json -o results/dance.json
```

输出 `dance.json` 含 fps、关节名、每帧世界根位置、xyzw 根四元数、按名称存储的关节角。`dance.stats.json` 含预处理/求解耗时和逐帧收敛诊断。`--frames 50` 限制输入帧数，`--stride 2` 下采样并同步更新输出 fps，`--pose t_pose` 选择旧预设归一化姿态；使用 JSON 时以 JSON 中的姿态为准。

JSON 配置定义 URDF、关键点、归一化关节姿态、脚尖、源语义关键点、源归一化 BVH；路径相对 JSON。可选 `edges` 是去重且 `i<j` 的边列表；省略时生成完整无向图。`ground` 必须位于 `key_joints` 最后。支持 `--unweighted`、`--no-normalize` 做消融。

迁移时可用 `--robot-preset FourierGR1Config.py` 读取旧版的字面量配置，使用 AST 提取而不执行旧代码；一般推荐显式 JSON。未知或非字面量配置会报错。

## 测试与性能实验

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/benchmark.py --project ../MAHR --frames 50
```

`benchmark.py` 比较相同数组目标函数的解析梯度与数值梯度，初值、逐帧边界、容差一致。可追加 `--legacy-graph`，调用旧版真实 `RobotGraphCallable`、graph distance 和 urdfpy FK；需要旧版依赖和 `../MAHR:../MAHR/third_party/urdfpy` 在 `PYTHONPATH` 上。`scripts/check_legacy.py` 使用同样的可选依赖验证源/目标图，覆盖 stand 和 T-pose。

**基准不含旧版首帧全局搜索**，三组均从相同归一化姿态做首帧局部优化，计时只覆盖求解。不应将该速度比表述为旧版默认完整流水线的速度比。新版默认将昂贵的全局初始化替换为局部初始化，属于明确的求解策略调整，可能影响极端首帧的局部最优质量。

本机实测 G1 / Clio 舞蹈 500 帧：求解约 4.84 秒，约 103 帧/秒，500 帧均报告收敛。这只是一个模型/动作的单次 CPU 测量，不代表通用性能或动作观感评价。对照测量见 [BENCHMARK.md](BENCHMARK.md)，逐帧数据见实际生成的 `results/*.stats.json`。

## 当前边界

已实现并验证 BVH → 源归一化关系数组 → 有界顺序优化 → 命名关节 JSON 的完整路径。当前版本针对平地、单自由度独立 URDF 关节，ASF/AMC、非平地高度图、mimic/free/planar 关节和可视化界面尚未实现；不宣称已替代原项目全部外围功能。新机器人的关节轴、根坐标约定和语义映射需要验证；8 种固定机器人均已完成 500 帧完整动作验证，汇总见 [ALL_ROBOTS.md](ALL_ROBOTS.md)。

不生成逐帧图或 pickle，不加载机器人网格，不计算动力学。各 Retargeter 使用独立 Robot 工作区，单实例不可并发调用。

## 离屏动作预览

需要系统 `ffmpeg` 和 OpenGL 离屏后端。当前机器可用软件 OSMesa：

```bash
MUJOCO_GL=osmesa python scripts/render_motion.py results/dance.json \
  --urdf ../MAHR/mahr/rsc/assets/urdf/unitree_g1/g1_29dof_stl.urdf \
  -o results/dance.mp4
```

默认每 4 帧渲染一次，并相应降低视频 fps，保持动作原速。可用 `--stride 1` 渲染全部帧。渲染保留视觉网格并重放根位姿和关节角，不执行动力学仿真。


## 全部机器人批量运行

```bash
PYTHONPATH=src python scripts/retarget_all.py --project ../MAHR --output results/all_robots
python scripts/render_all.py results/all_robots
python scripts/make_gallery.py results/all_robots
```

第一步包含每种机器人完整 500 帧求解、边界验证和首/中/尾帧解析梯度检查；可增加 `--check-legacy-fk` 调用旧 urdfpy 做 FK 位置对照（需要旧版依赖）。第二步生成每种机器人的真实网格视频，第三步生成带名称的八宫格视频、GIF 和 ZIP 结果包。

`configs/` 已提供八种机器人的 stand/T-pose JSON 配置。例如 `--config configs/atlas_stand.json`；每个配置明确列出源/目标关键点映射。


可通过 `--motion` 指定其他 BVH，输出到独立目录以保留历史结果：

```bash
PYTHONPATH=src python scripts/retarget_all.py --project ../MAHR \
  --motion ../MAHR/mahr/rsc/amass_bvh/smpl_h_small.bvh \
  --output results/all_robots_smpl_h_small
python scripts/render_all.py results/all_robots_smpl_h_small
python scripts/make_gallery.py results/all_robots_smpl_h_small
```

帧数、梯度抽查位置、视频帧数校验与结果包名称均按所选动作自动确定。
