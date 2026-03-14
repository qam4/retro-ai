"""Unit tests for SyntheticGenerator."""

import numpy as np
import torch

from retro_ai.training.simple import SyntheticGenerator, Transition, WorldModel


class _FakePolicy:
    """Minimal policy stub with SB3-compatible predict interface."""

    def __init__(self, num_actions: int = 4):
        self._num_actions = num_actions

    def predict(self, obs, deterministic=True):
        action = np.random.randint(0, self._num_actions)
        return action, None


def test_generate_returns_correct_number_of_transitions():
    """Each rollout produces exactly `horizon` transitions."""
    obs_shape = (4, 84, 84)
    num_actions = 4
    horizon = 5
    num_rollouts = 3

    model = WorldModel(obs_shape, num_actions)
    gen = SyntheticGenerator(model, horizon=horizon, device="cpu")
    policy = _FakePolicy(num_actions)

    start_obs = np.random.randint(0, 256, (num_rollouts, *obs_shape), dtype=np.uint8)
    transitions = gen.generate(start_obs, policy, num_rollouts)

    assert len(transitions) == num_rollouts * horizon


def test_generate_transition_fields():
    """Each transition has correct field types and shapes."""
    obs_shape = (4, 84, 84)
    num_actions = 4
    horizon = 3
    num_rollouts = 1

    model = WorldModel(obs_shape, num_actions)
    gen = SyntheticGenerator(model, horizon=horizon, device="cpu")
    policy = _FakePolicy(num_actions)

    start_obs = np.random.randint(0, 256, (num_rollouts, *obs_shape), dtype=np.uint8)
    transitions = gen.generate(start_obs, policy, num_rollouts)

    for t in transitions:
        assert isinstance(t, Transition)
        assert t.observation.shape == obs_shape
        assert t.observation.dtype == np.uint8
        assert t.next_observation.shape == obs_shape
        assert t.next_observation.dtype == np.uint8
        assert isinstance(t.action, int)
        assert isinstance(t.reward, float)
        assert isinstance(t.done, bool)


def test_generate_last_step_is_done():
    """The last transition in each rollout has done=True."""
    obs_shape = (4, 84, 84)
    num_actions = 4
    horizon = 4
    num_rollouts = 2

    model = WorldModel(obs_shape, num_actions)
    gen = SyntheticGenerator(model, horizon=horizon, device="cpu")
    policy = _FakePolicy(num_actions)

    start_obs = np.random.randint(0, 256, (num_rollouts, *obs_shape), dtype=np.uint8)
    transitions = gen.generate(start_obs, policy, num_rollouts)

    # Each rollout is exactly `horizon` steps, so transitions at indices
    # horizon-1, 2*horizon-1, etc. should have done=True
    for r in range(num_rollouts):
        last_idx = (r + 1) * horizon - 1
        assert transitions[last_idx].done is True
        # Non-last steps should have done=False
        for s in range(horizon - 1):
            idx = r * horizon + s
            assert transitions[idx].done is False


def test_generate_obs_chaining():
    """Each step's next_observation becomes the next step's observation."""
    obs_shape = (4, 84, 84)
    num_actions = 4
    horizon = 5
    num_rollouts = 1

    model = WorldModel(obs_shape, num_actions)
    gen = SyntheticGenerator(model, horizon=horizon, device="cpu")
    policy = _FakePolicy(num_actions)

    start_obs = np.random.randint(0, 256, (num_rollouts, *obs_shape), dtype=np.uint8)
    transitions = gen.generate(start_obs, policy, num_rollouts)

    # First obs should match start_obs
    np.testing.assert_array_equal(transitions[0].observation, start_obs[0])

    # Each subsequent obs should equal previous next_obs
    for i in range(1, len(transitions)):
        np.testing.assert_array_equal(
            transitions[i].observation, transitions[i - 1].next_observation
        )
