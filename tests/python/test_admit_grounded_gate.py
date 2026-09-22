"""The survival gate must reject a state that stays alive by BOUNCING, not by standing.

`admit_requires_survival` keeps a capture if the agent lives >= `min_survival_steps`
(30).
That cannot see a whole class of doomed state. Measured on L4: 81 of 100 `Low2_launch`
seeds sit on px 184 -- floor 12's tile edge, where the agent reads grounded for one
frame
and then falls (0/8 survive a NOOP hold, px 188 survives 8/8) -- and the trampoline
below
keeps them alive a MEDIAN OF 83 STEPS. So 100/100 doomed seeds were admitted and the
pool
meant to teach the rope-2 crossing taught the fall-bounce loop instead.

`admit_requires_grounded` adds the missing half: the capture must END its survival
window
in `SEED_POSES`. The escalator case is the reason that is SEED_POSES and not
SURFACE_POSES -- an earlier draft also required y to be unchanged, which rejected
25/25 of
L3's `Lesc_top` seeds, since pose 13 is a legitimate ride and an escalator carries the
agent down by design.
"""

import importlib.util
import pathlib

import pytest

_SRC = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "mo5"
    / "yeti"
    / "train_checkpoint_curriculum.py"
)


def _mod():
    spec = importlib.util.spec_from_file_location("_yeti_gate", _SRC)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - native deps absent
        pytest.skip(f"trainer module not importable here: {exc}")
    return mod


def _mgr(mod, **kw):
    return mod.CheckpointManager(
        max_states_per_checkpoint=10,
        min_states_to_advance=1,
        reset_fraction=0.0,
        frontier_fraction=0.0,
        earlier_fraction=0.0,
        min_survival_steps=30,
        **kw,
    )


FALL, DEATH, TRAMPOLINE_L, WALK, LADDER, ESCALATOR = 11, 12, 17, 0, 8, 13


def test_off_by_default_so_L1_L2_L3_are_unchanged():
    mod = _mod()
    m = _mgr(mod)
    assert m.admit_requires_grounded is False
    # a bouncing state still gets in, exactly as before
    assert m._admit_by_play(90, False, end_pose=TRAMPOLINE_L) == "survived"


def test_alive_but_bouncing_is_REJECTED_when_on():
    """The Low2_launch case: alive 90 steps (well past the 30-step window) because a
    trampoline caught it, but airborne when the window closed."""
    mod = _mod()
    m = _mgr(mod, admit_requires_survival=True, admit_requires_grounded=True)
    for pose in (FALL, DEATH, TRAMPOLINE_L):
        assert m._admit_by_play(90, False, end_pose=pose) == "rejected", pose


def test_alive_and_standing_is_KEPT_when_on():
    mod = _mod()
    m = _mgr(mod, admit_requires_survival=True, admit_requires_grounded=True)
    for pose in (WALK, LADDER):
        assert m._admit_by_play(90, False, end_pose=pose) == "survived", pose


def test_the_L3_ESCALATOR_RIDE_is_kept():
    """Pose 13 is a controlled descent the trainer deliberately seeds from. An earlier
    draft of this criterion rejected all 25 of L3's Lesc_top seeds."""
    mod = _mod()
    m = _mgr(mod, admit_requires_survival=True, admit_requires_grounded=True)
    assert ESCALATOR in mod.SEED_POSES
    assert m._admit_by_play(90, False, end_pose=ESCALATOR) == "survived"


def test_unknown_end_pose_falls_through_to_the_step_count():
    """None means the episode ended before the window closed, which the survived_steps
    test already handles -- the gate must not reject on missing information."""
    mod = _mod()
    m = _mgr(mod, admit_requires_survival=True, admit_requires_grounded=True)
    assert m._admit_by_play(90, False, end_pose=None) == "survived"
    assert m._admit_by_play(5, False, end_pose=None) == "rejected"


def test_a_short_lived_capture_is_still_rejected_regardless_of_pose():
    mod = _mod()
    m = _mgr(mod, admit_requires_survival=True, admit_requires_grounded=True)
    assert m._admit_by_play(5, False, end_pose=WALK) == "rejected"


def test_grounded_requirement_also_overrides_the_reached_next_shortcut():
    """With the flag on, ending the window airborne is disqualifying even if the episode
    made forward progress -- otherwise inherited credit would readmit the bounce states.
    """
    mod = _mod()
    m = _mgr(mod, admit_requires_grounded=True)
    assert m._admit_by_play(90, True, end_pose=WALK) == "reached"
    assert m._admit_by_play(90, True, end_pose=TRAMPOLINE_L) == "rejected"


# --- THE WIRING ------------------------------------------------------------------
#
# Everything above calls `_admit_by_play` directly and passes, because the gate's own
# logic was always right. What was never exercised is whether the CALLERS forward the
# pose they compute: `save_waypoint` and `save_scored` each take an `end_pose` argument,
# and each called `_admit_by_play(survived_steps, reached_next)` without it, so the
# parameter defaulted to None inside the gate and `admit_requires_grounded` could never
# fire. The env computes it (`_pose_at_window_end`) and passes it in at both call sites,
# the docstrings describe the criterion and quote its measured blast radius, and the one
# line that uses it was missing.
#
# Measured cost: L4 v22 ran 1M steps with the flag ON and rejected 238 of 1385
# `Low2_launch` captures -- 17.2%, against 17.9% per 1M for v21 with the flag OFF, i.e.
# no effect at all, while 35 of its 100 pooled seeds still sat on the lethal px 184.
# Under v22's own policy those seeds end the 30-step window airborne 28 times in 30.
#
# These tests go through the public entry points, so they fail if the forwarding is
# dropped again.


def test_save_waypoint_forwards_end_pose_to_the_gate():
    mod = _mod()
    m = _mgr(mod, admit_requires_survival=True, admit_requires_grounded=True)
    # Alive far beyond the window, but airborne on a trampoline when it closed.
    m.save_waypoint("Low2_launch", b"state", 90, False, end_pose=TRAMPOLINE_L)
    assert (
        "Low2_launch" not in m.waypoints
    ), "a bouncing capture was pooled: save_waypoint dropped end_pose"
    assert m.wp_rejected_precarious.get("Low2_launch") == 1


def test_save_waypoint_still_admits_a_grounded_capture():
    mod = _mod()
    m = _mgr(mod, admit_requires_survival=True, admit_requires_grounded=True)
    m.save_waypoint("Low2_launch", b"state", 90, False, end_pose=WALK)
    assert len(m.waypoints["Low2_launch"]) == 1


def test_save_scored_forwards_end_pose_to_the_gate():
    mod = _mod()
    m = _mgr(mod, admit_requires_survival=True, admit_requires_grounded=True)
    m.save_scored(1, b"state", 90, False, 0, 0, end_pose=TRAMPOLINE_L)
    assert (
        len(m.checkpoints[1]) == 0
    ), "a bouncing capture was pooled: save_scored dropped end_pose"
    assert m.stats["rejected_precarious"][1] == 1


def test_save_scored_still_admits_a_grounded_capture():
    mod = _mod()
    m = _mgr(mod, admit_requires_survival=True, admit_requires_grounded=True)
    m.save_scored(1, b"state", 90, False, 0, 0, end_pose=WALK)
    assert len(m.checkpoints[1]) == 1


def test_rejections_are_attributable_to_the_grounded_check():
    """Why this counter exists: the bug above was invisible for a whole run because the
    only rejection tally lumps every reason together, so a gate that never fires and a
    gate with nothing to reject read identically."""
    mod = _mod()
    m = _mgr(mod, admit_requires_survival=True, admit_requires_grounded=True)
    m.save_waypoint("Low2_launch", b"s1", 90, False, end_pose=TRAMPOLINE_L)  # airborne
    m.save_waypoint("Low2_launch", b"s2", 5, False, end_pose=WALK)  # too short
    assert m.wp_rejected_airborne.get("Low2_launch") == 1
    assert m.wp_rejected_precarious.get("Low2_launch") == 2
