from gym.envs.registration import register

register(
    id='dm_laikago-v0',
    entry_point='gym_ext_envs.gym_dm.envs:DMLaikagoEnv',
)