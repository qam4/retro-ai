"""`curriculum.seed_waypoint_skip` removes a waypoint from SEEDING only.

WHY IT EXISTS. L4 floor 11 (`Lclimb3_top`) is exposed -- no ceiling, on the kangaroo
path -- and holding NOOP there dies in 3-13 frames in every sampled state.
`admit_requires_survival` therefore admits only arrivals that landed in a benign hazard
phase, so the pool is measurably easier than reality: pool seeds survive a median of 8
NOOP frames against 3 for the policy's own arrivals from reset. Starting episodes there
hands the agent a survivable phase for free, and arriving survivably is the decision the
level turns on -- the approach to the floor-12 jump needs 8 steps, so a 3-frame arrival
is doomed whatever it does.

WHY A CONFIG AND NOT A MAP EDIT. `Target.seedable` is a property of the map, so clearing
it there changes every run at once and leaves no control arm reproducible from a config.
`mark_airborne` and `pay_on_target_change` were both added for exactly that reason.

WHAT MUST NOT MOVE. Only seeding. The reward keeps its own group list from
`LevelMap.reward_waypoints`, so an excluded waypoint is still a reward milestone and is
still detected; it simply stops being practice material.
"""

from __future__ import annotations

import pytest
from retro_ai.training.run_config import CurriculumConfig
from retro_ai.training.yeti_map import get_level_map


def test_default_is_empty_so_nothing_changes():
    """Pin the default: an opt-in lever must not alter a run that does not set it."""
    assert CurriculumConfig().seed_waypoint_skip == []


def test_it_accepts_a_list_of_names():
    c = CurriculumConfig(seed_waypoint_skip=["Lclimb3_top"])
    assert c.seed_waypoint_skip == ["Lclimb3_top"]


def test_two_configs_do_not_share_the_default_list():
    """`field(default_factory=list)` and not a bare `[]` -- a shared mutable default
    would make one run's exclusion leak into the next config built in the same process,
    which is exactly the kind of cross-run contamination that is invisible in a log."""
    a, b = CurriculumConfig(), CurriculumConfig()
    a.seed_waypoint_skip.append("Lclimb3_top")
    assert b.seed_waypoint_skip == []


def test_yaml_round_trip():
    import tempfile
    from pathlib import Path

    import yaml
    from retro_ai.training.run_config import RunConfig

    cfg = {
        "training": {"timesteps": 1000, "output": "out"},
        "env": {"profile": "yeti_fruit_level4"},
        "reward": {"name": "fruit_bonus_path_progress_pbrs_grounded", "params": {}},
        "curriculum": {"waypoints": True, "seed_waypoint_skip": ["Lclimb3_top"]},
    }
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.yaml"
        p.write_text(yaml.safe_dump(cfg))
        loaded = RunConfig.from_yaml(str(p))
    assert loaded.curriculum.seed_waypoint_skip == ["Lclimb3_top"]


@pytest.mark.parametrize("wp", ["Lclimb3_top"])
def test_excluding_a_waypoint_leaves_the_reward_untouched(wp):
    """The lever is about practice material, not about credit.

    If this ever fails, excluding a seeding spot has silently become a reward change and
    any A/B using it is measuring two things at once.
    """
    groups = get_level_map(4).reward_waypoints
    assert any(wp in g for g in groups), (
        f"{wp} is expected to remain a reward milestone; seed_waypoint_skip must not "
        "touch LevelMap.reward_waypoints"
    )
