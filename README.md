# 飞艇路径规划系统 (Airship Path Planning System)

基于强化学习的飞艇路径规划系统，结合风场数据同化技术，实现了在复杂风场环境下的智能路径规划。

## 项目结构

```
airship_pathplanning/
├── model.py           # 定义强化学习模型的特征提取器
├── Gauss.py           # 风场数据同化模块，基于高斯模型
├── train.py           # 模型训练和测试的主程序
├── conv_ppo.py        # 基于PPO算法的卷积神经网络路径规划实现
├── myenv_conv.py      # 路径规划环境(基础版)
└── myenv_conv_energy.py # 考虑能量消耗的路径规划环境
```

## 核心功能

1. **风场数据同化**：基于高斯模型融合预报风场与实时测量数据，提高风场预测精度
2. **深度强化学习**：使用PPO(Proximal Policy Optimization)算法训练路径规划模型
3. **多输入特征提取**：结合CNN(卷积神经网络)处理网格风场数据和向量特征
4. **路径可视化**：直观展示飞艇规划路径与风场环境的关系

## 关键模块说明

### 1. 风场数据同化 (Gauss.py)

实现了`WindFieldAssimilation`类，主要功能包括：
- 从NetCDF文件加载风场预报数据
- 提取以飞艇位置为中心的局部风场矩阵
- 使用高斯影响函数融合测量值与预报值
- 计算并可视化风场不确定性

```python
# 示例用法
assimilation = WindFieldAssimilation(window_size=32, f_path='wind_data.nc')
assimilation.update([88, 125], 0.5, 0.1)  # 更新风场
```

### 2. 特征提取器 (model.py)

`CustomCombinedExtractor`类实现了多模态特征融合：
- 使用CNN处理风场矩阵数据
- 结合向量特征(如位置、速度等)
- 通过全连接层输出融合特征

### 3. 模型训练与测试 (train.py & conv_ppo.py)

提供了完整的模型训练和测试流程：
- `train_and_save()`: 训练PPO模型并保存
- `runwithplot()`: 加载模型并可视化路径规划结果
- `run()`: 执行路径规划并返回轨迹数据

## 使用方法

### 训练模型

```python
# 训练考虑能量消耗的模型
train_and_save(PathPlanningEnv_energy, 32, "model/PPO_energyenv_conv32_500w", 5000000)
```

### 测试模型并可视化

```python
# 测试模型并显示路径
runwithplot("model/PPO_energyenv_conv32_500w", PathPlanningEnv_energy, 32)
```

## 依赖项

- Python 3.x
- PyTorch
- Stable Baselines3
- NumPy
- Matplotlib
- NetCDF4

## 备注

- 风场数据需以NetCDF格式提供，包含'u'和'v'变量
- 可通过调整`policy_kwargs`参数修改神经网络结构
- 环境大小(conv_size)可根据实际需求调整，默认32x32网格

该系统可应用于飞艇、无人机等在大气环境中运行的飞行器路径规划，通过考虑风场影响提高路径规划的效率和准确性。
