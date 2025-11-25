# Airship Path Planning System

A reinforcement learning-based airship path planning system that integrates wind field data assimilation technology to achieve intelligent path planning in complex wind field environments.

## Project Structure

```
airship_pathplanning/
├── model.py           # Defines the feature extractor for the reinforcement learning model
├── Gauss.py           # Wind field data assimilation module based on Gaussian model
├── train.py           # Main program for model training and testing
├── conv_ppo.py        # Convolutional neural network-based path planning implementation using PPO algorithm
├── myenv_conv.py      # Path planning environment (basic version)
└── myenv_conv_energy.py # Path planning environment considering energy consumption
```

## Core Features

1. **Wind Field Data Assimilation**: Fuses forecast wind field data with real-time measurement data based on a Gaussian model to improve wind field prediction accuracy
2. **Deep Reinforcement Learning**: Trains the path planning model using the PPO (Proximal Policy Optimization) algorithm
3. **Multi-Input Feature Extraction**: Combines CNN (Convolutional Neural Network) to process grid-based wind field data and vector features
4. **Path Visualization**: Intuitively displays the relationship between the planned airship path and the wind field environment

## Key Module Explanations

### 1. Wind Field Data Assimilation (Gauss.py)

Implements the `WindFieldAssimilation` class with core functionalities:
- Loads wind field forecast data from NetCDF files
- Extracts local wind field matrices centered at the airship's position
- Fuses measured values with forecast values using a Gaussian influence function
- Calculates and visualizes wind field uncertainty

```python
# Example Usage
assimilation = WindFieldAssimilation(window_size=32, f_path='wind_data.nc')
assimilation.update([88, 125], 0.5, 0.1)  # Update wind field
```

### 2. Feature Extractor (model.py)

The `CustomCombinedExtractor` class enables multi-modal feature fusion:
- Processes grid-based wind field data using CNN
- Integrates vector features (e.g., position, velocity, etc.)
- Outputs fused features through fully connected layers

### 3. Model Training & Testing (train.py & conv_ppo.py)

Provides a complete workflow for model training and testing:
- `train_and_save()`: Trains the PPO model and saves the checkpoint
- `runwithplot()`: Loads the trained model and visualizes path planning results
- `run()`: Executes path planning and returns trajectory data

## Usage Instructions

### Train the Model

```python
# Train model considering energy consumption
train_and_save(PathPlanningEnv_energy, 32, "model/PPO_energyenv_conv32_500w", 5000000)
```

### Test the Model & Visualize Results

```python
# Test model and display planned path
runwithplot("model/PPO_energyenv_conv32_500w", PathPlanningEnv_energy, 32)
```

## Dependencies

- Python 3.x
- PyTorch
- Stable Baselines3
- NumPy
- Matplotlib
- NetCDF4

## Notes

- Wind field data must be provided in NetCDF format, containing 'u' (zonal wind) and 'v' (meridional wind) variables
- The neural network structure can be modified by adjusting the `policy_kwargs` parameter
- The environment size (conv_size) can be adjusted according to actual requirements (default: 32x32 grid)

This system can be applied to path planning of airships, UAVs, and other aerial vehicles operating in atmospheric environments. By considering wind field effects, it improves the efficiency and accuracy of path planning.
