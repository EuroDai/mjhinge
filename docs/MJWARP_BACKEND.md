# MJWarp 后端

`mjhinge_warp` 提供独立于 C++ 动态库的批量仿真。实现方式参考 pc4 上的
`/home/rvsa/dzy/mujoco-cable-dynamics-warp`：读取原有 MJCF、移除原生插件声明、
保留配置元数据，并通过 Warp 内核和 MJWarp 回调实现物理行为。

## 安装

Python 3.10+，固定 MuJoCo / mujoco-warp 3.16.0、warp-lang 1.18.0。
不要在 C++ 插件使用的 MuJoCo 3.9 环境中升级这些依赖。

```bash
python3 -m venv .venv-warp
source .venv-warp/bin/activate
python -m pip install -r requirements-warp.txt
python -m pip install -e . --no-deps
```

Linux x86_64、CUDA 12 驱动（包括 pc4 的 NVIDIA R550）使用
`requirements-warp-cu12.txt`，安装固定 SHA256 的 Warp 官方 CUDA 12 wheel。
Windows 对应的激活命令为 `.venv-warp\Scripts\Activate.ps1`。

```bash
mjhinge-warp examples/hinge.xml --device cuda:0 --nworld 2048 --steps 1000 --graph \
  --output validation/mjwarp_cuda_hinge_2048.json
mjhinge-warp examples/hinge.xml --device cpu --steps 100
```

不安装包时可设置 `PYTHONPATH=python`，使用 `python -m mjhinge_warp.cli`。
CLI 输出不包含 JIT、预热、重置和图捕获的连续步进墙钟时间；计时前后同步设备。
简单铰链的吞吐不能代表完整手模型或策略训练速度。

## Python API

```python
from mjhinge_warp import HingeSimulation, load_xml

sim = HingeSimulation("examples/hinge.xml", nworld=256, device="cuda:0",
                      nconmax=64, njmax=256)
# XML 字符串也可以：HingeSimulation(load_xml(xml, base_dir="examples"), ...)
sim.step(5)
sim.reset()
sim.capture_step()
sim.step(100)
sim.forward()     # 更新动力学，不改变载荷历史
sim.reset([1, 3]) # 独立重置物理状态、外力和载荷历史
cpu_data = sim.get_data(0)
sim.check()
```

关键设备数组：

| 数组 | 形状 | 说明 |
| --- | --- | --- |
| `data.qpos` | `(nworld, nq)` | 关节位置 |
| `data.qvel` | `(nworld, nv)` | 关节速度 |
| `data.ctrl` | `(nworld, nu)` | 执行器控制 |
| `data.xfrc_applied` | `(nworld, nbody)`，元素为六维向量 | 世界坐标系 `[Fx,Fy,Fz,Tx,Ty,Tz]`，力矩参考刚体质心 |
| `loads` | `(nworld, nhinge)`，元素为二维向量 | 最近一次已求解步骤的 `[F_radial,F_axial]` |
| `limits` | `(nworld, nhinge)` | 最近一次动力学求解使用的摩擦上限 |

向量数组的 `.numpy()` 结果会多出末尾分量维度，例如 `loads.numpy()` 是
`(nworld, nhinge, 2)`。实例顺序为 `sim.loaded.hinges`，包含名称、joint/dof/body ID。
`sim.set_control(vector_or_batch)` 可上传控制向量；更高效的设备训练循环直接写入
`data.ctrl`。模型原生 `dof_frictionloss` 保留在 `sim.cpu_model` 中，不被逐步修改。

图捕获前至少预热一次。捕获后保持数组地址不变，使用 `.assign()` 或设备内核原位
更新。局部重置和控制更新可以与已有图一起使用；修改回调、物理选项、拓扑后先调用
`clear_graph()`，再重新捕获。`capture_step()` 本身不推进 CUDA 仿真时间。
`check()` 会同步设备，适合调试和批次边界；普通 `step()` 不读取 CPU 数值。

## 求解顺序

1. 为活跃实例预留摩擦约束行，包括原生 `frictionloss=0` 的关节。
2. 在 passive 回调中按当前阶段的位置、速度和上一物理步载荷计算上限，替换
   `data.efc.frictionloss`。预留行的极小正数只用于约束构建，求解前会替换为实际值，
   包括零值。各环境有独立约束数据。
3. 原生求解器处理摩擦、接触和等式约束，不叠加显式摩擦力矩。
4. 启用 MJWarp 的 `run_rne_postconstraint`，在积分修改速度之前计算完整反力。
5. 物理步结束后才推进一次载荷历史。RK4 每阶段刷新摩擦上限，最终保存最后一个
   动力学阶段的反力，与 C++ 后端约定一致。

MuJoCo 3.16 的 RNE 已恢复原生空间肌腱力，因此不能重复使用 3.9 C++ 后端的
端点补偿。原生肌腱载荷由 MJWarp 的 RNE 计算。

## 支持范围

- 现有七个数值参数和原有基础摩擦；每刚体唯一 hinge，每关节唯一实例。
- Euler、implicit、implicitfast、RK4；Newton / CG 与 dense / sparse 的接触、等式测试。
- 重力、世界坐标系外力、子树反力、原生 tendon（含 pulley 和执行器力限幅）。
- CPU / CUDA、多环境、独立重置、CUDA 图、前向查询。
- MJCF include、相对于主文件的资源目录；递归 include 会明确报错。

尚不支持外部引擎插件、`body_force_sensors`、sleeping body，均明确拒绝加载。
与参考 cable Warp 后端的混合模型还需要单独的回调/刚体力接入，当前不会自动组合。
其他原生功能受 MJWarp 3.16 自身支持范围限制。

必须调用 `HingeSimulation.step()`，直接调用 `mjw.step(model, data)` 不会推进
独立载荷历史。`get_data()` 用于检查/显示；载荷历史由 `loads` 单独保存，
不能仅通过导出的原生 `MjData` 恢复完整仿真。后端使用 float32，不保证与 C++ double
逐位一致。基础约束软化与原模型一致，低于摩擦阈值时仍可能有少量数值滑移。

## 验证

```bash
MJHINGE_TEST_DEVICE=cuda:0 python -m pytest tests/test_mjwarp_backend.py -q
MJHINGE_TEST_DEVICE=cpu python -m pytest tests/test_mjwarp_backend.py -q
```

测试包含加载/卸载的一步延迟、零基础摩擦、多环境、静止保持/启动、耗散方向、
角度增益、低速增强、四种积分器、重力/子树/旋转轴、步前速度采样、肌腱和限幅、
接触/等式、局部重置、CUDA 图控制更新与重置、解析减速/停止。

跨版本 C++ 轨迹对照分两步。先在 MuJoCo 3.9 的环境导出，不需要安装 Warp：

```bash
python tests/compare_mjwarp_reference.py --model examples/hinge.xml \
  --plugin build/mjhinge.so --steps 1000 --export validation/mjwarp_reference_hinge.npz
```

Windows 对应库路径为 `build-x64/Release/mjhinge.dll`。然后在 Warp 环境比较：

```bash
python tests/compare_mjwarp_reference.py --model examples/hinge.xml \
  --reference validation/mjwarp_reference_hinge.npz --device cuda:0 \
  --output validation/mjwarp_compare_hinge_cuda.json
```

脚本逐步检查 qpos、qvel、qacc 和载荷，预先固定 float32 绝对误差阈值并在失败时
返回非零退出码。该对照只验证保存的模型与初始状态，不表示任意接触轨迹都等价。

`--same-state` 另外逐点恢复 C++ 轨迹的位置、速度和上一步载荷，再比较加速度。
这可以区分连续 float32 积分产生的状态误差与相同状态下的动力学误差；不会改变
原有连续轨迹判定，也不会将其失败改成通过。

## pc4 实测（2026-10-30）

测试目录为 `/home/rvsa/dzy/mjhinge`，GPU 为 RTX 4090，驱动 550.163.01。
此目录的 `.venv-warp` 是独立环境，使用 Warp 1.18.0+cu12。可直接运行：

```bash
cd ~/dzy/mjhinge
MJHINGE_TEST_DEVICE=cuda:0 .venv-warp/bin/python -m pytest tests/test_mjwarp_backend.py -q
.venv-warp/bin/mjhinge-warp examples/hinge.xml --device cuda:0 \
  --nworld 2048 --steps 1000 --graph --output validation/mjwarp_cuda_hinge_2048.json
```

本地 MuJoCo 3.9 C++ 插件重新构建后，原有 18 项测试通过。pc4 最终独立环境
CUDA **25 项通过**；CPU **24 项通过、1 项 CUDA 专用测试跳过**。
pc4 创建环境时 GitHub wheel 下载超时，
因此从指定参考仓库已验证的本地安装复制运行时包，再离线安装本仓库；没有修改
参考仓库环境。标准安装仍使用上面的固定版本及 SHA256 依赖文件。

1000 步的 C++ 3.9 / CUDA 3.16 对照：

| 模型 | 最大位置误差 rad | 最大速度误差 rad/s | 最大载荷误差 N | 最大瞬时加速度误差 rad/s² | 严格连续轨迹判定 |
| --- | --- | --- | --- | --- | --- |
| `hinge.xml` | 2.92e-6 | 1.50e-5 | 2.46e-5 | 8.35e-4 | 通过 |

简单铰链的 2048 环境、1000 步 CUDA 图连续运行耗时约 0.195 s，合计约
1051 万环境步/秒；不包含 JIT、预热和显示。该结果只对应单铰链，不代表
完整强化学习训练吞吐。完整记录见 `validation/mjwarp_cuda_hinge_2048.json`。
