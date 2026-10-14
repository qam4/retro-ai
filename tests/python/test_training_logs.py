"""The training logs report where episodes start and how each start scores.

Fruits seed into pools named for them (F1..F4) like every other route target, so
nothing has filled a rung slot (`checkpoints[1..N]`) since. Three outputs kept
indexing by rung slot anyway, and on L1 v19 they described slots nothing fills while
the agent reached F4 from the level start half the time:

    status line         success=[0->1:100%, 1->2:N/A, ...]  gscore=[0.47, 0.00, ...]
    curriculum_diag.csv succ_ema1-4, gscore1-4 always 0; start_frac0 = 1.000 while
                        2128 of 7325 starts (29%) were from the level start
    admission_diag.csv  every column 0

These pin the replacement: the level start and every pool are reported by the key
episodes.csv uses, a rung slot appears only while it holds states, and the level
start's share is out of ALL starts.

No emulator needed.
"""

import csv
import importlib.util
import pathlib
import random
import re

import pytest

_SRC = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "mo5"
    / "yeti"
    / "train_checkpoint_curriculum.py"
)

FRUITS = [frozenset({f"F{i}"}) for i in (1, 2, 3, 4)]
WALK = 0  # a grounded pose: passes the admission gate's pose check


@pytest.fixture(scope="module")
def tcc():
    spec = importlib.util.spec_from_file_location("_yeti_training_logs", _SRC)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - native deps absent
        pytest.skip(f"trainer module not importable here: {exc}")
    return mod


def _mgr(tcc):
    return tcc.CheckpointManager(
        max_states_per_checkpoint=10,
        min_states_to_advance=1,
        reset_fraction=0.0,
        frontier_fraction=0.0,
        earlier_fraction=0.0,
        min_survival_steps=30,
        mandatory_groups=FRUITS,
        n_rungs=len(FRUITS),
        admit_requires_survival=True,
    )


def _seeded(tcc):
    """A manager whose fruit pools hold states, as on L1 after the first few hundred
    thousand steps."""
    m = _mgr(tcc)
    for i in (1, 2, 3, 4):
        for k in range(3):
            m.save_waypoint(f"F{i}", f"F{i}-{k}".encode(), 90, False, end_pose=WALK)
    return m


def _play(m, n, seed=0):
    """Draw ``n`` starts the way training does; return the start keys drawn."""
    random.seed(seed)
    keys = [m.pick_start()[0] for _ in range(n)]
    assert {0, "F1"} <= set(keys), "the draw must mix the level start and pools"
    return keys


def _callback(tcc, m, tmp_path):
    cb = tcc.CurriculumCallback(
        1_000_000,
        diag_path=str(tmp_path / "curriculum_diag.csv"),
        n_rungs=m.N_RUNGS,
        manager=m,
    )
    return cb


def _rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def test_the_status_line_carries_no_array_indexed_by_rung_slot(tcc):
    """reset_reach is per rung but is fed by every level-start episode, so it always
    has data. Any other per-slot array is the bug this file exists for."""
    m = _seeded(tcc)
    for k in _play(m, 200):
        m.record_episode(k, 2 if k == 0 else 3)
    s = m.summary()
    assert re.findall(r"(\w+)=\[", s) == ["reset_reach"], s
    assert f"level_start_score={m.checkpoints[0].goal_score:.2f}" in s, s
    assert m.checkpoints[0].goal_score > 0, "the level start was scored"


def test_a_rung_slot_is_reported_only_while_it_holds_states(tcc):
    m = _seeded(tcc)
    assert [k for k, _p, _n in m.start_sources()] == [0, "F1", "F2", "F3", "F4"]
    assert "rung" not in m.route_table()
    # seed_archive (and an inherited pre-name-pool file) can still fill one.
    m.save_checkpoint(2, b"archived", source_cp=2)
    assert [k for k, _p, _n in m.start_sources()][:2] == [0, 2]
    assert "rung2" in [line.split()[0] for line in m.route_table().splitlines()]
    assert "seeds=13 in 5 pools" in m.summary(), "a filled rung slot holds seeds"


def test_the_route_table_shows_each_pools_score_and_share(tcc):
    m = _seeded(tcc)
    m.checkpoints[0].goal_score = 0.47
    m.waypoints["F3"].goal_score = 0.25
    table = m.route_table(shares={0: 0.30, "F3": 0.70})
    head, *body = table.splitlines()
    cols = head.split()
    rows = {line.split()[0]: dict(zip(cols, line.split())) for line in body}
    assert body[0].split()[0] == "level-start", "the level start leads the table"
    assert rows["level-start"]["score"] == "0.47"
    assert rows["level-start"]["share"] == "0.30"
    assert rows["F3"]["score"] == "0.25"
    assert rows["F3"]["share"] == "0.70"
    assert rows["F2"]["share"] == "—", "no share given reads as no data, not 0"


def test_level_start_share_is_out_of_every_start(tcc, tmp_path):
    """The v19 case: most starts come from named pools. The old start_frac0 divided
    by rung-slot starts only, so it read 1.000 here."""
    m = _seeded(tcc)
    keys = _play(m, 400)
    cb = _callback(tcc, m, tmp_path)
    cb.num_timesteps = 5000
    cb._write_diag()
    (row,) = _rows(tmp_path / "curriculum_diag.csv")
    expected = keys.count(0) / len(keys)
    assert 0.05 < expected < 0.95, "the draw must not be all one kind"
    assert float(row["level_start_share"]) == pytest.approx(expected, abs=5e-4)
    assert not any(c.startswith(("succ_ema", "start_frac", "gscore")) for c in row)
    assert {"reach1", "reach4", "reach_princess"} <= set(row), "retrospective reads"


def test_pool_diag_counts_starts_per_pool_per_interval(tcc, tmp_path):
    m = _seeded(tcc)
    cb = _callback(tcc, m, tmp_path)
    first = _play(m, 300, seed=1)
    cb.num_timesteps = 5000
    cb._write_diag()
    second = _play(m, 100, seed=2)
    cb.num_timesteps = 10000
    cb._write_diag()
    rows = _rows(tmp_path / "pool_diag.csv")
    for step, keys in (("5000", first), ("10000", second)):
        mine = {r["pool"]: r for r in rows if r["step"] == step}
        for key in set(keys):
            label = "level-start" if key == 0 else key
            assert int(mine[label]["starts"]) == keys.count(key), (step, key)
        assert sum(float(r["share"]) for r in mine.values()) == pytest.approx(
            1.0, abs=5e-3
        )
    assert not (tmp_path / "admission_diag.csv").exists()


def test_pool_diag_reports_admissions_per_interval_and_refused_targets(tcc, tmp_path):
    """A target whose every capture is refused has no pool, and still needs a row:
    "everything captured here dies" is what the admission columns are for."""
    m = _seeded(tcc)
    cb = _callback(tcc, m, tmp_path)
    m.save_waypoint("L45_top", b"doomed", 5, False, end_pose=WALK)
    cb.num_timesteps = 5000
    cb._write_diag()
    m.save_waypoint("F1", b"F1-new", 90, False, end_pose=WALK)
    cb.num_timesteps = 10000
    cb._write_diag()
    rows = {(r["step"], r["pool"]): r for r in _rows(tmp_path / "pool_diag.csv")}
    refused = rows[("5000", "L45_top")]
    assert (refused["size"], refused["rejected"], refused["score"]) == ("0", "1", "")
    assert rows[("5000", "F1")]["admitted_survived"] == "3", "this run's captures"
    assert rows[("10000", "F1")]["admitted_survived"] == "1", "an interval delta"
    assert rows[("10000", "L45_top")]["rejected"] == "0"
    start = rows[("5000", "level-start")]
    assert (start["size"], start["admitted_reached"], start["rejected"]) == ("", "", "")


def test_a_start_group_gets_its_own_pool_diag(tcc, tmp_path):
    m = _seeded(tcc)
    cb = tcc.CurriculumCallback(
        1_000_000,
        diag_path=str(tmp_path / "curriculum_diag_L5.csv"),
        n_rungs=m.N_RUNGS,
        manager=m,
    )
    cb.num_timesteps = 5000
    cb._write_diag()
    assert (tmp_path / "pool_diag_L5.csv").exists()
    assert not (tmp_path / "pool_diag.csv").exists(), "the L1 group's file is apart"


def test_an_interval_with_no_starts_has_no_shares(tcc, tmp_path):
    m = _seeded(tcc)
    cb = _callback(tcc, m, tmp_path)
    cb.num_timesteps = 5000
    cb._write_diag()
    (row,) = _rows(tmp_path / "curriculum_diag.csv")
    assert row["level_start_share"] == ""
    assert all(r["share"] == "" for r in _rows(tmp_path / "pool_diag.csv"))
    assert tcc._interval_shares({0: 4, "F1": 6}, {0: 4, "F1": 6}) is None
