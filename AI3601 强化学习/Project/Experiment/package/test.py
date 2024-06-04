from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs
import numpy as np

env = suite.load("walker", "walk")
env.reset()
action = np.random.random(6)
time_step = env.step(action)
for k, v in time_step.observation.items():
	print("observation:", k, v.shape)
print("original reward:", time_step.reward)

state = env.physics.get_state()
print("physics:", state.shape)
print("-----")
##########

new_env = suite.load("walker", "walk")
reward_spec = new_env.reward_spec()
new_env.reset()
with new_env.physics.reset_context():
	new_env.physics.set_state(state)
new_reward = new_env.task.get_reward(new_env.physics)    # 输入是 env 当前的状态 physics, 通过 env.task.get_reward 函数输出奖励
new_reward = np.full(reward_spec.shape, new_reward, reward_spec.dtype)
print("new reward:", new_reward)

# states = episode['physics']
#     for i in range(states.shape[0]):
#         with env.physics.reset_context():
#             env.physics.set_state(states[i])
#         reward = env.task.get_reward(env.physics)    # 输入是 env 当前的状态 physics, 通过 env.task.get_reward 函数输出奖励
#         reward = np.full(reward_spec.shape, reward, reward_spec.dtype)  # 改变shape和dtype
#         rewards.append(reward)
#     episode['reward'] = np.array(rewards, dtype=reward_spec.dtype)
#     return episode
