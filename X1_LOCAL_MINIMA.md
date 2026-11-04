# X1 多初值与局部解验证

## 结果

存在初值依赖的异常局部收敛分支，但不能据此把当前完整流程的全部异常归因于求解器陷入局部极小值。

在原限位的第 350 帧，固定同一个几何目标、关节限位、源姿态及根旋转，关闭时序项和逐帧步长边界。从异常分支继续优化，损失停在 0.06509317，足部参考倾角约 140.84°；从正常参考分支优化，损失为 0.06413121，倾角约 29.21°。后者低约 1.48%，说明同一几何目标下确有更低损失的非翻转解，而异常初值未找到它。严格容差复核后分支仍存在，投影梯度残差均低于 4.3e-6；这支持局部收敛分支判断，不构成严格 Hessian 局部极小证明。

在原限位第 85 帧，找到的异常解反而比非翻转参考分支损失更低（0.04261630 vs 0.07700379）。在双踝双轴 ±45° 的第 250 帧，多初值改善损失（0.08050326 -> 0.07362000），足部参考倾角仍约 110°。因此优化分支、目标函数和时序历史需要共同考虑，单纯更换初值或加严容差不能保证恢复自然姿态。

## 控制变量

两套限位：当前原限位和双踝 pitch/roll 均 ±45°。每套检查 85、120、250、350、450、490 帧（0-based）。每个目标从当前解、前一帧解、仅关键点地面项生成的参考解、历史正常解及 6 个固定种子扰动初值求解。参考姿态并非正常姿态的人工标签；最终足方向仍独立计算。

三种实验共 360 次求解：

- production_step_bounds：与当前帧完全相同的时序历史、目标项及逐帧可行域。所有初值先投影到同一边界，因此不同参考在很窄的可行域内可能变得接近；不能把它当广泛全局搜索。
- joint_bounds_only：只移除逐帧步长边界，保留原历史及全部损失项。
- geometry_only：移除逐帧边界和时序损失，保留图匹配、原姿态正则、关键点和网格地面惩罚及关节限位。此项改变了目标，结论只能用于几何子问题。

生产边界下，检查的 12 帧多初值最低损失均只发生微小改变，没有找到从翻转变为非翻转且损失更低的结果。放开步长但保留时序历史后，检查的异常帧仍未找到非翻转解。这不是不存在正常解的证明：时序项会惩罚离开已有异常历史，有限初值也无法穷尽所有分支。

所有 360 次求解报告成功；初轮使用生产容差。36 个最佳解处全维中心差分验证，解析梯度最大绝对误差为 1.72e-08，未发现梯度实现错误的证据。生产容差下成功停止不代表已达到很小的梯度残差，因此另对三个代表场景的九个结果进行严格复核（ftol=1e-14、gtol=1e-8、maxiter=10000、maxls=80）。

## 严格容差复核

| 限位 | 帧 | 初始分支 | 最终损失 | 足部参考倾角 ° | 投影梯度无穷范数 |
|---|---:|---|---:|---:|---:|
| baseline | 85 | current | 0.04449388 | 143.36 | 2.07e-06 |
| baseline | 85 | marker_reference | 0.07700379 | 38.86 | 8.67e-07 |
| baseline | 85 | historical_reference | 0.04261630 | 143.38 | 1.56e-06 |
| baseline | 350 | current | 0.06509317 | 140.84 | 4.25e-06 |
| baseline | 350 | marker_reference | 0.06413121 | 29.21 | 3.59e-07 |
| baseline | 350 | historical_reference | 0.06413121 | 29.21 | 2.05e-07 |
| pitch45_roll45 | 250 | current | 0.08050326 | 112.84 | 9.88e-07 |
| pitch45_roll45 | 250 | marker_reference | 0.07362000 | 110.02 | 2.08e-06 |
| pitch45_roll45 | 250 | historical_reference | 0.07362000 | 110.02 | 9.15e-07 |

投影梯度定义为 `||x - clip(x - gradient, lower, upper)||_inf`。足部倾角沿用前一轮静止足部向上参考轴定义，取两足最大值；不是接触标签，也不保证小于 90° 的姿态在所有方面自然。这里不宣称找到了全局最优。

## 逐组数据

| 限位 | 帧 | 实验 | 当前损失 | 最佳损失 | 当前倾角 ° | 最佳倾角 ° |
|---|---:|---|---:|---:|---:|---:|
| baseline | 85 | production_step_bounds | 0.0971406 | 0.0971406 | 117.0 | 117.0 |
| baseline | 85 | joint_bounds_only | 0.0971406 | 0.0954295 | 117.0 | 121.4 |
| baseline | 85 | geometry_only | 0.0968510 | 0.0426167 | 117.0 | 143.4 |
| baseline | 120 | production_step_bounds | 0.0739199 | 0.0739199 | 84.7 | 84.7 |
| baseline | 120 | joint_bounds_only | 0.0739199 | 0.0739199 | 84.7 | 84.8 |
| baseline | 120 | geometry_only | 0.0734875 | 0.0641123 | 84.7 | 84.1 |
| baseline | 250 | production_step_bounds | 0.0544500 | 0.0544500 | 146.6 | 146.6 |
| baseline | 250 | joint_bounds_only | 0.0544500 | 0.0544500 | 146.6 | 146.6 |
| baseline | 250 | geometry_only | 0.0541512 | 0.0318951 | 146.6 | 140.1 |
| baseline | 350 | production_step_bounds | 0.0672547 | 0.0672547 | 127.3 | 127.3 |
| baseline | 350 | joint_bounds_only | 0.0672547 | 0.0672547 | 127.3 | 127.3 |
| baseline | 350 | geometry_only | 0.0669849 | 0.0641319 | 127.3 | 29.1 |
| baseline | 450 | production_step_bounds | 0.0761493 | 0.0761493 | 122.1 | 122.1 |
| baseline | 450 | joint_bounds_only | 0.0761493 | 0.0760966 | 122.1 | 123.1 |
| baseline | 450 | geometry_only | 0.0758229 | 0.0539802 | 122.1 | 143.0 |
| baseline | 490 | production_step_bounds | 0.0472770 | 0.0472769 | 120.9 | 120.9 |
| baseline | 490 | joint_bounds_only | 0.0472770 | 0.0472672 | 120.9 | 120.9 |
| baseline | 490 | geometry_only | 0.0467832 | 0.0444577 | 120.9 | 124.0 |
| pitch45_roll45 | 85 | production_step_bounds | 0.1077805 | 0.1077804 | 36.3 | 36.2 |
| pitch45_roll45 | 85 | joint_bounds_only | 0.1077805 | 0.1019213 | 36.3 | 33.6 |
| pitch45_roll45 | 85 | geometry_only | 0.1072746 | 0.0770052 | 36.3 | 38.6 |
| pitch45_roll45 | 120 | production_step_bounds | 0.0663688 | 0.0663688 | 41.9 | 41.9 |
| pitch45_roll45 | 120 | joint_bounds_only | 0.0663688 | 0.0663688 | 41.9 | 41.9 |
| pitch45_roll45 | 120 | geometry_only | 0.0660690 | 0.0641536 | 41.9 | 44.1 |
| pitch45_roll45 | 250 | production_step_bounds | 0.0832289 | 0.0832289 | 118.1 | 118.1 |
| pitch45_roll45 | 250 | joint_bounds_only | 0.0832289 | 0.0832159 | 118.1 | 118.0 |
| pitch45_roll45 | 250 | geometry_only | 0.0825817 | 0.0736221 | 118.1 | 110.0 |
| pitch45_roll45 | 350 | production_step_bounds | 0.0912712 | 0.0912712 | 72.9 | 72.9 |
| pitch45_roll45 | 350 | joint_bounds_only | 0.0912712 | 0.0900530 | 72.9 | 70.4 |
| pitch45_roll45 | 350 | geometry_only | 0.0904866 | 0.0641318 | 72.9 | 29.2 |
| pitch45_roll45 | 450 | production_step_bounds | 0.0724531 | 0.0724531 | 52.7 | 52.7 |
| pitch45_roll45 | 450 | joint_bounds_only | 0.0724531 | 0.0724531 | 52.7 | 52.7 |
| pitch45_roll45 | 450 | geometry_only | 0.0722182 | 0.0689759 | 52.7 | 37.9 |
| pitch45_roll45 | 490 | production_step_bounds | 0.0881801 | 0.0881801 | 60.7 | 60.7 |
| pitch45_roll45 | 490 | joint_bounds_only | 0.0881801 | 0.0862342 | 60.7 | 67.2 |
| pitch45_roll45 | 490 | geometry_only | 0.0877316 | 0.0722285 | 60.7 | 89.5 |

复现（上层工作目录）：

```bash
PYTHONPATH=MAHR_v2/src python MAHR_v2/scripts/check_x1_local_minima.py
PYTHONPATH=MAHR_v2/src python MAHR_v2/scripts/check_x1_local_minima_strict.py
```

所有初值分支的最终坐标、损失分量、停止信息和残差保存在 `results/x1_local_minima/summary.json` 与 `strict_probe.json`。本次没有改变生产求解器或默认参数。


完整 marker_reference 几何分支轨迹已扩展到 500 帧，见 `results/x1_reference_branch/x1.json` 与同目录 `x1.mp4`；脚本为 `scripts/retarget_x1_reference_branch.py`。第 350 帧精确复现该表对应结果，但整段仍有 260 帧参考足倾角超过 90°，并有独立帧优化造成的不连续，不能将其称为全程正常轨迹。
