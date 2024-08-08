"""Wrapper for flattening observations of an environment."""
import gym
import gymnasium
import gym.spaces as spaces
import numpy as np

# modify the class from gym.wrappers.FlattenObservation
class CustomizedFlattenObservation(gym.ObservationWrapper):
    """Observation wrapper that flattens the observation.

    Example:
        >>> import gym
        >>> env = gym.make('CarRacing-v1')
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> env = FlattenObservation(env)
        >>> env.observation_space.shape
        (27648,)
        >>> obs = env.reset()
        >>> obs.shape
        (27648,)
    """

    def __init__(self, env: gym.Env):
        """Flattens the observations of an environment.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.flattened_obs_space = self._custom_flatten_space(env.observation_space)
        self.observation_space = spaces.flatten_space(self.flattened_obs_space)

    def observation(self, observation):
        """Flattens an observation.

        Args:
            observation: The observation space to flatten

        Returns:
            The flattened observation space
        """

        flattened_obs = self._custom_flatten_ob(observation)
         
        return spaces.flatten(self.flattened_obs_space, flattened_obs)
    
    def _custom_flatten_ob(self, observation):
        # flatten the observation
        # handle the nested dictionary structure
        def flatten_dict_ob(d, parent_key='', sep='_'):
            items = {}
            for k, v in d.items():
                new_key = parent_key + sep + str(k) if parent_key else k
                if isinstance(v, dict):
                    items.update(flatten_dict_ob(v, new_key, sep=sep))
                else:
                    items[new_key] = v
            return items
        
        return flatten_dict_ob(observation)
        
    
    def _custom_flatten_space(self, observation_space):
        # flatten the observation
        # handle the nested dictionary structure
        def flatten_dict_space(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = parent_key + sep + str(k) if parent_key else k
                if isinstance(v, gymnasium.spaces.dict.Dict):
                    items.extend(flatten_dict_space(v, new_key, sep=sep).items())
                else:
                    if isinstance(v, gymnasium.spaces.box.Box):
                        # change the dtype
                        # It seems that spaces.flatten_space() can not deal with 'Float64' type.
                        # Convert the `dType` of observation space to 'Float32'
                        new_box = spaces.Box(
                            low=v.low,
                            high=v.high,
                            shape=v.shape,
                            dtype=np.float32  # Change to desired dtype
                        )
                        items.append((new_key, new_box))
                    else:
                        raise NotImplementedError("The type of observation is not supported.")
            return spaces.Dict(items)
        
        return flatten_dict_space(observation_space)
        

