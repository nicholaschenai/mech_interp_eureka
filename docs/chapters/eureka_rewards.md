# Eureka Rewards
The below sections show the evolution of the reward function via Eureka.

Each section is titled with its Eureka iteration, model nickname (strong, medium, weak), and the qualitative result of the trained model.

In each section, we show
- The outcome of the previous iteration (if available)
- The LLM's feedback on the previous iteration (if available)
- The LLM's thoughts on deciding the reward function
- The reward function for the current iteration

## Iter 0 (weak): Reaches handle but doesn't open door

To define a reward function for the task of opening the cabinet door, we focus on variables related to the position and movement of the agent's hand (Franka robot) and the cabinet door. The objective is to incentivize the robot's hand to reach and pull the cabinet handle effectively. Here's how the reward function can be structured:

```python
import torch
from typing import Tuple, Dict

@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward component for minimizing the distance between the robot hand and the drawer handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    dist_to_handle_reward = -distance_to_handle

    # Reward component for opening the cabinet door (measured by the joint position of the door)
    door_opening_reward = cabinet_dof_pos[:, 3]  # Assuming index 3 corresponds to the door position

    # Reward component for velocity of the door opening (positive for opening direction)
    door_velocity_reward = cabinet_dof_vel[:, 3]  # Assuming index 3 corresponds to the door velocity

    # Transform the distance reward using a temperature parameter
    temperature_distance = 0.1  # This value can be tuned
    transformed_dist_reward = torch.exp(dist_to_handle_reward / temperature_distance)

    # Combine all reward components
    total_reward = 0.5 * transformed_dist_reward + 1.0 * door_opening_reward + 0.5 * door_velocity_reward

    # Collect the individual components in a dictionary
    reward_components = {
        "distance_to_handle": dist_to_handle_reward,
        "door_opening": door_opening_reward,
        "door_velocity": door_velocity_reward,
        "transformed_distance_reward": transformed_dist_reward
    }

    return total_reward, reward_components
```

This reward function focuses on three main aspects:
1. **Distance to handle**: By reducing the distance between the robot's hand and the drawer handle, we incentivize the robot to reach towards the handle.
2. **Door opening**: Rewarding the position of the door joint when it opens increases the incentive to successfully open the door.
3. **Door velocity**: Encourages the movement of the door in a positive direction to ensure it is opening.

The distance reward is transformed using `torch.exp` with a temperature parameter to moderate its influence. This transformation can help to handle diminishing returns as the robot reaches closer to the target point.

## Iter 1 (medium): Reaches handle and opens door, but not wide enough to complete task

```
Metrics from previous iteration:
distance_to_handle: ['-0.47', '-0.01', '-0.01', '-0.01', '-0.01', '-0.35', '-0.01', '-0.55', '-0.01', '-0.02'], Max: -0.01, Mean: -0.08, Min: -1.48 
door_opening: ['0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00'], Max: 0.03, Mean: 0.00, Min: 0.00 
door_velocity: ['0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00'], Max: 0.03, Mean: 0.00, Min: -0.01 
transformed_distance_reward: ['0.01', '0.88', '0.90', '0.89', '0.88', '0.04', '0.89', '0.06', '0.86', '0.86'], Max: 0.90, Mean: 0.75, Min: 0.00 
task_score: ['0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00'], Max: 0.01, Mean: 0.00, Min: 0.00 
episode_lengths: ['312.00', '500.00', '500.00', '500.00', '500.00', '500.00', '500.00', '500.00', '500.00', '500.00'], Max: 500.00, Mean: 497.95, Min: 312.00 
```
Analyzing the policy feedback indicates several key issues with the current reward function:

1. **Distance to Handle Reward**: The values are quite low (negative) and stable, indicating that this component dominates due to the transformation applied and the reward structure not effectively guiding the hand to the handle.

2. **Door Opening and Velocity Rewards**: Both components show negligible influence, as their values are consistently zero. This suggests these components are ineffective in their current form, perhaps because the agent doesn't get a sufficient reward to reach and pull the handle.

3. **Transformed Distance Reward**: While more variable, the values are still relatively low, illustrating a minimal incentive for the policy towards optimal action.

4. **Overall Policy Performance**: The nearly zero task success rate and long episode lengths indicate that the agent struggles to make progress on the task.

To improve the reward function, it would be beneficial to:

- **Redesign** the reward components to better align with the task objectives.
- **Rescale** or **disregard** ineffective components.
- **Apply transformations** where necessary with appropriate temperature scaling.

Here's a revised reward function incorporating these considerations:

```python
import torch
from typing import Tuple, Dict

@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward component for minimizing the distance between the robot hand and the drawer handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.2  # Adjusted temperature parameter for scaling
    dist_reward = torch.exp(-distance_to_handle / temperature_distance)

    # Reward for any motion that indicates the door is opening
    opening_restored = cabinet_dof_pos[:, 3].clamp(min=0.0)
    opening_restored_reward = opening_restored * 2.0  # Multiply to scale the effect

    # Velocity indicating door movement in the opening direction
    velocity_reward = torch.clamp(cabinet_dof_vel[:, 3], min=0.0) * 1.5  # Adjusted reward scaling

    # Combine all reward components
    total_reward = 0.5 * dist_reward + 1.0 * opening_restored_reward + 1.0 * velocity_reward

    # Collect each individual component in a dictionary
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
```

### Key Changes
1. **Redesigned Distance Component**: Instead of minimizing distance directly, it’s transformed with an adjusted temperature, providing a more nuanced and trailing reward as the hand approaches the drawer.

2. **Scaled Door Rewards**: Both door opening and velocity rewards are more heavily weighted and importantly focus not only on positive movement (i.e., actually opening the door) but also rescaling to emphasize tangible movements along the task objective. 

Overall, the goal of these modifications is to incentivize precise movements that align with achieving the task effectively. As always, real-world tuning may still be necessary to find optimal temperature and scaling parameters.

## Iter 2 (strong): Reaches handle and opens door, completes task
```
Metrics from previous iteration:
dist_reward: ['0.11', '0.55', '0.53', '0.47', '0.29', '0.54', '0.44', '0.44', '0.23', '0.47'], Max: 0.74, Mean: 0.46, Min: 0.09 
opening_restored_reward: ['0.00', '0.63', '0.64', '0.62', '0.61', '0.58', '0.67', '0.67', '0.65', '0.62'], Max: 0.70, Mean: 0.62, Min: 0.00 
velocity_reward: ['0.00', '0.03', '0.03', '0.07', '0.06', '0.07', '0.07', '0.07', '0.07', '0.09'], Max: 0.60, Mean: 0.07, Min: 0.00 
task_score: ['0.00', '0.09', '0.05', '0.07', '0.06', '0.04', '0.05', '0.03', '0.04', '0.01'], Max: 0.44, Mean: 0.06, Min: 0.00 
episode_lengths: ['223.00', '480.06', '479.05', '482.33', '493.88', '487.66', '491.94', '495.69', '488.67', '491.44'], Max: 500.00, Mean: 475.70, Min: 125.12 
```

From the policy feedback, it's evident that several issues with the reward function are present:

1. **Distance Reward**: The values are low and don't vary much, indicating the current transformation does not effectively prioritize reducing distance between the hand and the handle.

2. **Opening Restored Reward**: While showing higher values, they are relatively stable, hinting that door opening is somewhat achieved but with likely insufficient encouragement for further exploration or precision in opening.

3. **Velocity Reward**: These values are minimal and don't vary significantly, indicating this component doesn't provide significant benefit in its current form.

4. **Task Score and Episode Lengths**: Task success remains low, with episodes being overly long, suggesting inefficient learning or policy execution.

To address these issues, I will redesign and re-implement the reward components to better guide the policy towards successful task completion. I will provide an entirely new reward function with adjusted transformations and scaling:

```python
import torch
from typing import Tuple, Dict

@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Improved reward for reducing distance to the handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.1  # Reduced temperature to increase sensitivity
    dist_reward = torch.exp(-distance_to_handle / temperature_distance)

    # Enhanced reward for achieving and maintaining door opening
    door_open_value = cabinet_dof_pos[:, 3].clamp(min=0.0)  # Focus on the positive (opening)
    opening_restored_reward = torch.tanh(door_open_value) * 3.0  # Emphasize through tanh transformation

    # Amplifying the velocity reward for consistent door movement direction
    door_velocity = cabinet_dof_vel[:, 3]
    temperature_velocity = 0.1
    velocity_reward = torch.exp(door_velocity / temperature_velocity) * 0.5  # Adjusted scaling for balance

    # Compose the total reward
    total_reward = 1.0 * dist_reward + 2.0 * opening_restored_reward + 0.5 * velocity_reward

    # Dictionary with individual components for detailed analysis
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
```

### Key Improvements:

1. **Revised Distance Reward**: I reduced the temperature to make the reward more sensitive to changes in proximity, thereby promoting finer grasp precision.

2. **Enhanced Opening Reward**: Switched to a `tanh` transformation, maximizing the reward when the door is fully open while still encouraging substantial opening with an increased multiplier.

3. **Amplified Velocity Reward**: Applied an `exp` transformation to prioritize consistent velocity while encouraging steady door movement, with balanced scaling.

These changes aim to offer stronger and more appropriately scaled incentives for each aspect of the task, potentially leading to improved learning outcomes and higher task success rates.