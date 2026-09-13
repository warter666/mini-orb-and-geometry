# 03-vision — 纯 numpy 的 ORB 特征 + 几何视觉

参照 ORB_SLAM2 与《视觉 SLAM 十四讲》，不调用任何计算机视觉库的函数，全部 numpy 实现。
两个包：特征管线 `mini_orb` + 几何视觉 `mini_geom`。

## mini_orb：特征管线

- **FAST-9** 角点：16 像素圆周向量化（`sliding_window_view`），圆周加倍处理跨零连续段
- **Harris 打分**：梯度结构张量 + 积分图盒滤波，贪心半径-5 非极大值抑制
- **强度质心主方向**：半径 15 圆形邻域，`atan2(m01, m10)`
- **steered BRIEF**：256 对采样点限制在半径 14 圆内（防旋转出界），5×5 盒滤波平滑，
  按关键点方向旋转后比较，输出 32 字节描述子
- **匹配**：`np.bitwise_count` 汉明距离 + 互最近邻 + Lowe 比率检验

## mini_geom：几何视觉

| 文件 | 内容 | 关键点 |
|------|------|--------|
| `geometry.py` | Rodrigues、Umeyama 配准、投影/去畸变、数值雅可比、LM | 阻尼按对角缩放，自适应接受 |
| `calib.py` | 张正友平面标定 | 归一化 DLT → 闭式内参 → 外参 → 联合 LM（含 k1,k2） |
| `pnp.py` | DLT / P3P（Sylvester 结式）/ LM 精化 / RANSAC | P3P 返回多解，用第 4 点消歧 |
| `flow.py` | 金字塔 LK + 6 参数仿射 LK | 每步在当前 warp 重建 Hessian，回溯保证 SSD 下降 |

```bash
python -m mini_orb.test_features      # 特征测试（3 项）
python -m mini_geom.test_minigeom     # 几何测试（11 项）
python -m demo_pipeline               # 端到端：渲染→ORB→匹配→RANSAC-PnP
```

## 实测（全部合成真值自校验，无外部数据）

```
标定 0.3px 噪声      fx 误差 0.186%  主点 0.43px  k1 -0.1802(真值 -0.18)  rms 0.276px
去畸变后重投影       0.0000 px（原始 47.29 px）
DLT+LM PnP 0.5px     rms 5.27 → 0.45 px   旋转误差 0.24°
P3P 3 点无噪声       4 个候选，最优解旋转误差 0°
RANSAC 30% 粗差      56/80 内点，旋转误差 0.094°（普通最小二乘 179°）
金字塔 LK 12.6px     单层 15.87 px → 4 层 0.026 px
仿射 LK 5°/1.06×     中位相对误差 0.0030，12 点全部优于 5%
端到端 demo          匹配内点率 94%，RANSAC 195/208，位姿误差 0.094° / 0.0007
```

## 已知边界（诚实记录）

- 仿射 LK 单层收敛域约 1° 旋转；>1° 必须靠金字塔，且需要真正的二维结构纹理——
  带限白噪声的自相关峰太窄，六参数拟合会漂到退化（收缩）解。
- P3P 本质多解：3 点最多 4 组位姿都能精确重投影，生产用法必须配合第 4 点或 RANSAC。
- 平面目标使 DLT-PnP 秩亏；平面场景应走 P3P + RANSAC，或改用单应分解求位姿。
- 金字塔 LK 的边界约束是固有的：点距图像边缘需 ≥ `min_safe_margin(levels, win)`，
  否则粗层跳过、只能停在初始值。
