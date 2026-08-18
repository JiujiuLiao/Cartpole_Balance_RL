# 倒立摆 PID 调参状态记录

**控制参数最后更新**：2026-06-02

**视觉链路覆盖更新**：2026-08-18

> 这是当前调参的工作状态，供下次开新会话时快速 onboard。
> 只记录控制相关信息，不涉及硬件诊断历史。
>
> 视觉覆盖说明：当前发布器是
> `src/vision/camerapreview_detect_yellow_tape_line_V1.py`，端口 9999。
> 它输出模 180° 的黄色胶带直线方向，不能区分竖直向上/向下，
> `pole_up` 恒为 0。自动能量甩起当前禁用；只允许已人工确认向上的
> isolated balance，以及围绕已知竖直平衡点的小幅测量。

---

## 1. 当前架构

- **架构**：单进程，Pi 上跑。swing-up 调 `compute_pwm`，catch 触发后**内联调用** `balance_loop()`（同进程，无切换延迟）。balance fall 后自动回 swing-up loop。
- **控制频率**：swing-up 50 Hz，balance 100 Hz
- **视觉**：yellow-tape-line V1（端口 9999、9 字段兼容 CSV），控制器用 `camera_comm.py` 订阅

---

## 2. 关键文件

| 文件 | 角色 |
|---|---|
| `cartpole-pid/src/swing_up_energy_dc.py` | swing-up + 内联 balance（**核心调参文件**）|
| `cartpole-pid/src/config.py` | 全局常量 + balance 增益 Kp/Kd/Kx/signs |
| `src/vision/camerapreview_detect_yellow_tape_line_V1.py` | 当前黄色胶带直线视觉发布器 |
| `cartpole-pid/dc_motor_controller/dc_motor_controller.ino` | ESP32 固件（已稳定，不用动）|

---

## 3. 当前参数（多次迭代后的工作值）

### config.py — balance PD 增益

```python
Kp = 40.0        # 历史：80（饱和）→ 10（过弱）→ 40（当前）
Kd = 5.0
Kx = 0.02        # 0 → 0.02 防 cart 漂；可考虑试 0.05
ANGLE_SIGN = -1  # 这套硬件 +PWM=cart 右；balance 需要 -1 才方向对
OMEGA_SIGN = +1
CART_SIGN  = +1
PWM_DEADBAND = 22
GIVEUP_ANGLE_DEG = 25.0
PWM_LIMIT = 255
```

### swing_up_energy_dc.py — swing-up + catch

```python
SWING_PWM       = 255
CATCH_ANGLE_DEG = 10.0   # 历史：25→15→8→10；8° 时摆杆够不着，10° 是经验阈值
CATCH_OMEGA_MAX = 0.5    # 历史：2→1→0.5；越紧 balance 越好接但 catch 越难触发
CTRL_HZ         = 50
center_correction 系数 = 0.40  # 历史：0.12 → 0.40，cart 漂明显减少
```

---

## 4. 能量公式（**已修过 2 次，保持现状**）

```python
E_diff = 0.5 * omega**2 + (GRAVITY/PENDULUM_L_M) * (cos(theta) - 1)
u      = -E_diff * omega * cos(theta)
# E_diff < 0 → 泵能；> 0 → 自动抽能
# 负号是这套硬件 +PWM=cart 右的方向约定，empirical 验证过
```

**老公式 bug（千万别改回去）**：
`E = ω² - (2g/L)cos(θ)` 永远 ≥ 0 → 过顶不抽能 → ω 越甩越大。新公式自动 drain，是必须保留的修复。

---

## 5. 已观察到的关键行为模式

- **swing-up 能甩到顶**：约 10-20 秒能让 \|θ\| < 10° 一次
- **能 catch**：catch 条件触发频率合理（数秒到十几秒一次）
- **balance 接管后撑不住**：通常 200-500 ms 内 fall。摆杆从 catch 状态（小 θ、小 ω）冲到 \|θ\|>25° 触发 giveup
- **cart 漂移**：长时间运行 cart 单向累积漂移，可能撞边界（centering 系数从 0.12→0.40 后改善但未根除）
- **catch 时摆 ω 偏大**：catch 阈值 0.5 时摆经常踩着 0.4-0.5 进 balance，给 balance 留时间不够

---

## 6. 还没试过的调参方向（下次优化候选）

| 方向 | 具体改动 | 预期效果 |
|---|---|---|
| 降低 SWING_PWM | 255 → 180-200 | 减少每周期能量注入，摆杆温和到顶，ω 更小 |
| 再收紧 catch ω | 0.5 → 0.3 或更小 | balance 接手时摆几乎不动；但 catch 可能永远不触发 |
| 加大 Kd | 5 → 8 或 10 | balance 阻尼增强，对抗摆杆角速度 |
| 加 catch hold time | 要求条件持续 N ms | 过滤瞬时巧合，确保真稳了再接手 |
| 加大 Kx | 0.02 → 0.05 | cart 漂更快被拉回中心 |
| 调 ω 低通滤波 | OMEGA_LPF_ALPHA 当前 0.35 | 对噪声敏感度可调 |

---

## 7. 下次开新窗口的开场建议

> "我继续调倒立摆 balance 调参。当前已经能 swing-up + 触发 catch，但 balance 撑不住几百毫秒。请读取 `C:\Projects\Inverted_Pendulum\cartpole-pid\TUNING_STATE.md` 了解当前参数和已观察行为，然后我想试 [具体方向]"
