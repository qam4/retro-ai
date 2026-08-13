"""Tests for the chain parsers in ``scripts/mo5/yeti/smoke_train.py``.

The smoke test's whole job is to FAIL when a warm-started run loses the
from-reset chain. That only works if the parsers read the manager's log lines
correctly, so those are pinned here against real line shapes. No emulator, no
training — pure text.
"""

import importlib.util
import pathlib

import pytest

_SMOKE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "mo5"
    / "yeti"
    / "smoke_train.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("smoke_train", _SMOKE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


smoke = _load()


# A real line from yeti_ctrlA_refactor_600k/output.log, trimmed of the middle
# arrays that the parsers do not read.
CTRL_A_LINE = (
    "step 600000/600000 (100%) | reward=N/A | emu_fps=2625 | "
    "reset_reach=[1.00, 1.00, 1.00, 0.96, 0.95, 0.95, 0.93, 0.77, 0.00, 0.00, "
    "0.00, 0.00, 0.00, 0.00, 0.00] | route[19]: 7/19 reached>=0.5 from reset"
)

# v14, whose from-reset chain collapsed.
V14_LINE = (
    "step 500000/15000000 (3%) | reward=21.2 | "
    "reset_reach=[1.00, 0.98, 0.97, 0.95, 0.00, 0.00, 0.00] | "
    "route[19]: 3/19 reached>=0.5 from reset"
)


def test_parse_chain_reads_route_scalar():
    assert smoke.parse_chain(CTRL_A_LINE) == (7, 19)


def test_parse_chain_takes_the_last_occurrence():
    """Progress lines repeat; the verdict must come from the newest one."""
    text = "\n".join([V14_LINE, CTRL_A_LINE])
    assert smoke.parse_chain(text) == (7, 19)


def test_parse_chain_absent():
    assert smoke.parse_chain("step 1000/600000 | reward=N/A") is None


def test_parse_reset_reach():
    reach = smoke.parse_reset_reach(V14_LINE)
    assert reach == [1.00, 0.98, 0.97, 0.95, 0.00, 0.00, 0.00]


def test_parse_reset_reach_takes_the_last_occurrence():
    text = "\n".join([V14_LINE, CTRL_A_LINE])
    assert len(smoke.parse_reset_reach(text)) == 15


def test_parse_reset_reach_absent():
    assert smoke.parse_reset_reach("no arrays here") is None


@pytest.mark.parametrize(
    "reach,expected",
    [
        ([1.0, 1.0, 1.0, 0.96, 0.95, 0.95, 0.93, 0.77, 0.0], 7),
        ([1.0, 0.98, 0.97, 0.95, 0.0, 0.0], 3),
        # Rung 1 already below threshold: nothing holds past the reset.
        ([1.0, 0.10, 0.0], 0),
        # An isolated high value ABOVE a break is not part of the chain.
        ([1.0, 0.9, 0.0, 0.99, 0.99], 1),
        ([1.0], 0),
    ],
)
def test_deepest_rung_is_contiguous(reach, expected):
    assert smoke.deepest_rung(reach) == expected


def test_deepest_rung_threshold_is_configurable():
    reach = [1.0, 0.6, 0.6, 0.2]
    assert smoke.deepest_rung(reach, threshold=0.5) == 2
    assert smoke.deepest_rung(reach, threshold=0.7) == 0


def test_report_passes_on_an_intact_chain(capsys):
    assert smoke.report(CTRL_A_LINE, min_chain=6, min_depth=None) == 0
    assert "PASS" in capsys.readouterr().out


def test_report_fails_on_a_collapsed_chain(capsys):
    assert smoke.report(V14_LINE, min_chain=6, min_depth=None) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "3/19" in out


def test_report_leaves_rungs_informational_by_default(capsys):
    """Rung INDICES moved when the ladder re-keyed (L3 went 3 rungs -> 15), so a
    depth assertion would wrongly fail a pre-refactor log. Default is off."""
    pre_refactor = (
        "reset_reach=[1.00, 0.00, 0.00] | route[19]: 7/19 reached>=0.5 from reset"
    )
    assert smoke.report(pre_refactor, min_chain=6, min_depth=None) == 0
    assert "informational" in capsys.readouterr().out
    # Opt in and the same log fails.
    assert smoke.report(pre_refactor, min_chain=6, min_depth=4) == 1


def test_report_fails_when_the_run_never_printed_a_route_scalar(capsys):
    assert (
        smoke.report("step 1000/600000 | reward=N/A", min_chain=1, min_depth=None) == 1
    )
    assert "no route scalar" in capsys.readouterr().out
