try:
    import numpy as np

    if not hasattr(np, "Inf"):
        np.Inf = np.inf
except Exception:
    pass

try:
    import gym
    from gym import error as gym_error
    from gym.envs import registration

    _original_register = registration.register

    def _register_allow_existing(id, *args, **kwargs):
        try:
            return _original_register(id, *args, **kwargs)
        except gym_error.Error as exc:
            if "Cannot re-register id" in str(exc):
                return None
            raise

    registration.register = _register_allow_existing
    gym.envs.registration.register = _register_allow_existing
except Exception:
    pass
