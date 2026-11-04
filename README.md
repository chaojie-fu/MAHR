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

## 时间连续性

默认在求解时约束关节速度为 8 rad/s（本仓库机器人均为转动关节）、根高度速度为 2 m/s，并加入相邻姿态差与二阶差分正则。约束使用输出 fps，支持下采样；这些是可调的重定向参数。没有对输出做后处理滤波。

可通过 `--max-joint-velocity`、`--temporal-weight`、`--acceleration-weight` 调整。`--legacy-temporal` 关闭新增时序项和速度约束，用于旧目标函数对照。`stats.json` 分别报告总目标、原始图误差、速度和加速度。修复前后的测量见 [CONTINUITY.md](CONTINUITY.md)。下述早期性能和全机器人结果是修复前的历史测量；`benchmark.py` 显式使用旧时序设置保证对照目标一致。

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

不生成逐帧图或 pickle，仅在初始化时加载脚部 STL 并缓存凸包支撑点，不计算动力学。各 Retargeter 使用独立 Robot 工作区，单实例不可并发调用。

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


## 归一化边长下限与地面净空

默认源端、机器人端分别使用 `max(rest_edge_length, 0.05 * rest_skeleton_height)` 作分母。身高估计为归一化姿态中最高关节/刚体原点减去最低脚尖高度；不包含视觉网格外壳。源和机器人使用各自单位，比例保持无量纲。`--min-edge-ratio 0.03` 可改为 3%；设为 `0` 显式恢复旧分母（源端 `length+1e-5`，机器人零长取 1）。`--no-normalize` 则不做归一化。

源地面边先按世界竖直净空定义，将负净空截到 0，避免以源动作的穿透作为目标。目标端保留有符号净空；显式启用 `--ground-weight` 时独立添加 `max(0,-h)^2` 单侧惩罚，保证穿透时仍有向上的梯度。惩罚包括 ground 关键点、URDF 碰撞球底面以及脚部 STL 凸包顶点；图边本身不截断。网格顶点惩罚按每个脚部网格取平均，避免网格密度增加权重；顶点梯度聚合到刚体 Jacobian 后计算，不构建大规模逐顶点 Jacobian。

`--ground-weight` 当前默认 `0`，防穿地惩罚不参与默认目标；显式设为非零时，代价除以目标身体高度平方；`--allow-source-ground-penetration` 关闭源端净空截断；`--no-foot-surfaces` 关闭脚部 STL 支撑点，仅保留关键点/碰撞球。脚部通过 toe 到最近含 mesh 的祖先 link 定位，当前支持 STL。只检查这些脚部表面及显式关键点/球体，不保证全身视觉网格无穿透。

若需完整旧目标对照，组合使用 `--legacy-temporal --posture-weight 0 --foot-support-weight 0 --min-edge-ratio 0 --ground-weight 0 --allow-source-ground-penetration --no-foot-surfaces --no-source-ground-calibration`。早期基准与旧代码回归脚本已显式保持旧设置。当前结果和限制见 [NORMALIZATION_GROUND.md](NORMALIZATION_GROUND.md)。


## 源地面校准（默认开启）

读取整段 BVH，在每帧取左右脚关键点的最低 z，再取这些高度的 5% 分位作为一个静态地面平面。全段所有关节的 z 同减此值，随后才构造地面边和截断负净空；不改变水平位移、根姿态或帧间竖直运动。不根据机器人分别估计，也不逐帧调整地面。

校准始终在 `--frames`/`--stride` 采样之前使用完整输入动作估计，避免仅改变采样范围就得到不同地面。CLI 和 `retarget_all.py` 默认行为一致：

- `--no-source-ground-calibration` 关闭校准，恢复输入零平面。
- `--source-ground-height VALUE` 指定已知地面高度，单位与原 BVH 长度相同，可为负。
- `--source-ground-quantile 0.05` 修改自动估计分位。

校准模式、原始地面高度、估计帧数和单位写入 `stats.json` 的 `normalization.source_ground`。自动估计是缺少地面标注时的启发式；整段腾空、异常关键点或无落地帧时应手动给定地面。

校准修复了 iCub Dance 的主要地面目标冲突，但没有解决 X1 的足部翻转，详见 [SOURCE_GROUND.md](SOURCE_GROUND.md)。历史结果不会自动替换；新全机器人轨迹保存为 `results/calibrated_all_dance/` 与 `results/calibrated_all_smpl/`。


当前默认已移除防穿地惩罚和近地脚底倒置惩罚（对应权重均为 0），保留不饱和静止姿态先验、时序项及源地面校准。八机器人两段动作对照见 [NO_GROUND_ABLATION.md](NO_GROUND_ABLATION.md)，新结果为 `results/no_ground_no_support_dance/`、`results/no_ground_no_support_smpl/`。此前近地约束实验见 [FOOT_SUPPORT.md](FOOT_SUPPORT.md)。
