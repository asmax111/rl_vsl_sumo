from gym_sumo.envs.sumo_env import SUMOEnv
from gym_sumo.envs.SUMOInitializeEnv import SUMOEnv_Initializer
from gym.envs.registration import register

register(
    id='SumoGUI-v0',
    entry_point='gym_sumo.envs:SUMOEnv_Initializer',
)


