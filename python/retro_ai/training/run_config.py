"""Typed training-run configuration.

Every training run is specified by a single YAML file. This module defines
the nested dataclass schema, loads YAML into it, and round-trips back to
a plain dict for manifest persistence.

Design principles
-----------------

1. **One YAML, no inheritance.** Every value either has a dataclass default
   or is required. Keeps resolution trivial; no mystery "this field came
   from somewhere else".

2. **Fail-loud on unknown keys.** A typo in a config should raise, not
   silently use the default. Helps catch stale configs after refactors.

3. **Round-trippable.** ``RunConfig.from_yaml(p).to_dict()`` yields the
   same structure the YAML describes (plus resolved defaults). That dict
   is what the run manifest persists, so replaying a run never needs the
   original YAML.

Typical use
-----------

>>> cfg = RunConfig.from_yaml("experiments/003-yeti/configs/segment_1to2.yaml")
>>> cfg.training.timesteps
5000000
>>> cfg.reward.name
'fruit_bonus'
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingConfig:
    """Top-level run controls."""

    timesteps: int
    output: str
    num_envs: int = 8
    seed: Optional[int] = None
    resume: Optional[str] = None
    # Save a model snapshot every N timesteps (None -> 2,000,000). Useful
    # for capturing a transient peak in a run that later degrades.
    snapshot_freq_steps: Optional[int] = None
    # How often to print the route table (one row per route point with its
    # from-reset reach). That table is the view you actually read to see how far
    # the agent gets and where it stops, so it defaults to often. None => 50k.
    route_table_freq_steps: Optional[int] = None
    # Phase-2 warm-start: load only the network WEIGHTS from ``resume``
    # and keep this run's PPO hyperparameters (n_steps, target_kl, ...).
    # A full PPO.load would instead restore the checkpoint's saved
    # hyperparameters, silently ignoring the new ones. Default False
    # preserves the original full-state resume behavior.
    warmstart_weights_only: bool = False
    # Where to inherit the curriculum's seed pools from. Default (None) is
    # ``dirname(resume)/checkpoints.pkl``, which assumes the weights and the pools sit
    # in the same directory. They often do not: a keep-best sweep writes the champion to
    # ``<run>/best/best_model.zip`` while the pools stay at ``<run>/checkpoints.pkl``,
    # so resuming from a champion silently inherits NO pools -- the load just misses.
    # Set this to the pool file explicitly when warm-starting from a champion or a
    # snapshot.
    resume_pools: Optional[str] = None


@dataclass(frozen=True)
class EnvConfig:
    """Environment wiring.

    Most fields default to ``None`` meaning "use whatever the game profile
    specifies". That way a config only has to mention what it overrides.
    """

    profile: str
    action_mode: str = "joystick"
    max_steps: int = 1000
    stall_threshold: int = 15
    resize: Optional[Tuple[int, int]] = (84, 84)
    frame_skip: Optional[int] = None
    frame_stack: Optional[int] = None
    frame_maxpool: Optional[bool] = None
    grayscale: Optional[bool] = None
    # (y, x, height, width), applied to the raw frame BEFORE grayscale/resize.
    # ``None`` = use the game profile's ``crop`` (itself usually None = no crop).
    # Use it to drop chrome the agent should not see, e.g. a HUD strip: pixels
    # that carry no navigational information but DO change when the emulator's
    # renderer changes, which silently invalidates trained policies (see the
    # `2b0a45d` blocker in TODO.md). Cropping changes the observation geometry,
    # so a policy trained with one crop cannot be evaluated with another.
    crop: Optional[Tuple[int, int, int, int]] = None


@dataclass(frozen=True)
class PPOConfig:
    """Stable-Baselines3 PPO hyperparameters.

    ``n_steps = None`` signals "compute from num_envs" downstream
    (typically ``max(1, 128 // num_envs)`` to keep rollout buffer size
    roughly constant).
    """

    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: Optional[int] = None
    n_epochs: int = 4
    ent_coef: float = 0.01
    clip_range: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # Optional KL budget per update (SB3 early-stops the epoch loop once
    # approx_kl exceeds it). None = no cap (default). Lowering it tames
    # oversized policy updates — see experiment 003 H-V (phase-2 anneal).
    target_kl: Optional[float] = None


@dataclass(frozen=True)
class RewardConfig:
    """Reference to a named reward formula + its parameters."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CurriculumConfig:
    """Settings used only by ``train_checkpoint_curriculum.py``."""

    reset_fraction: float = 0.4
    frontier_fraction: float = 0.4
    earlier_fraction: float = 0.2
    max_states_per_checkpoint: int = 100
    min_states_to_advance: int = 20
    seed_archive: Optional[str] = None
    # Optional path to a checkpoints-format pool (.pkl with
    # {"checkpoints": [list per CP]}) used to pre-seed the curriculum
    # pools with diverse, segment-derived start states. Levels 1..3
    # are loaded; CP4 is intentionally skipped so the frontier sits at
    # CP3 (the current wall) rather than jumping past it.
    preseed_pool: Optional[str] = None
    # Reach gate (approach 31): a CP level is only eligible as a start
    # state once the agent reaches it *from reset* at least this
    # often (EMA). Below it, the level's pool is off-distribution
    # garbage (rare lucky reaches), so we don't waste budget there.
    # This makes the curriculum advance one wall at a time, always
    # training the deepest reset-reachable segment on on-distribution
    # seeds.
    reach_threshold: float = 0.15
    # Anti-starvation floor: fraction of the non-reset (curriculum)
    # allocation spread UNIFORMLY across eligible segments, blended with
    # the (1 - success) weighting in pick_start. 0.0 = pure weighting.
    # >0 guarantees each eligible segment a minimum share so a very-hard
    # segment (CP4->princess, ~2% success) can't starve its prerequisite
    # (CP3->CP4) and destabilize reach-4.
    segment_floor: float = 0.0
    # MultiInputPolicy: when true, the observation is a Dict of the
    # image plus a 4-d fruit-presence vector, and the PPO policy is
    # "MultiInputPolicy" instead of "CnnPolicy". This de-aliases
    # checkpoint start states (a reset state vs "F1 collected" look
    # near-identical at 84x84) so a single policy can attach different
    # actions to them — the fix for the v6 composition wall. Note: a
    # run with this enabled cannot warm-start from a CnnPolicy model
    # (different observation space / network).
    multi_input_obs: bool = False
    # Minimum number of frames the agent must survive *after* a
    # checkpoint snapshot, under its own policy, for that state to be
    # admitted to the seed pool. Replaces the old passive-noop probe
    # (state_validator) which rejected ~99% of real mid-action
    # pickups. A snapshot that led to the next checkpoint in the same
    # episode is always admitted regardless of this threshold.
    # NOTE: this counts GYM STEPS (each = frame_skip emulator frames), not
    # emulator frames — hence the name (was `min_survival_frames`, which was
    # misleading). Legacy configs using `min_survival_frames` are aliased in
    # from_dict().
    min_survival_steps: int = 30
    # --- Level awareness (defaults preserve level-1 behavior) ---
    # Total collectible fruits in the level (level 1 = 4, level 2 = 2).
    # Drives CP indexing, the fruit-presence vector dim, and the
    # princess goal index (= fruits_total + 1).
    fruits_total: int = 4
    # Per-fruit presence RAM addresses (byte != 0 = on map, 0 =
    # collected). ``None`` -> the level-1 default
    # {1:0x2FAD, 2:0x2F00, 3:0x2E68, 4:0x2DD8}. These are *positional*
    # tilemap cells, so they differ per level (see
    # experiments/003-yeti/ram_map_re.md). YAML maps int->int, e.g.
    # ``{1: 11950, 2: 11975}`` for level 2 (0x2EAE, 0x2EC7).
    fruit_presence_addrs: Optional[Dict[int, int]] = None
    # Path to a save-state loaded on a CP0 (reset) start, instead of a
    # fresh game reset. ``None`` -> game reset (= level 1 start). Level 2
    # uses ``output/mo5/yeti/level2/level2_start.sav``.
    start_state: Optional[str] = None
    # WAYPOINT curriculum (H-AI). When true, the env captures grounded
    # states at computed ladder-top/bottom positions (yeti.waypoints) into
    # optional, non-gating start-pools, and pick_start samples them in the
    # same self-regulating goal-score draw as CPs. Off by default (L1 and
    # non-WP L2 runs are byte-identical). ``waypoint_tolerance`` is the
    # per-axis RAM-x/pixel-y window for "reached a waypoint".
    waypoints: bool = False
    # HOW THIS NUMBER SHOULD BE CHOSEN (measured on MO5 Yeti L4, 2026-08-24;
    # debug/l4_climb_shot.py and the per-step delta scan in level4_notes.md).
    #
    # It is ONE number applied to BOTH axes, but the axes are in different units:
    # x is the RAM byte 0x2B52 in 4-PIXEL units, y is the RAM byte 0x2B51 in
    # PIXELS. So `waypoint_tolerance: 2` means +-8 px horizontally and +-2 px
    # vertically -- a 4x asymmetry that nobody chose on purpose.
    #
    # What the window actually has to cover. Detection is pose-gated
    # (`pose in SEED_POSES`), so only surface/ladder poses can ever be marked.
    # Measured |delta| per GYM STEP (= frame_skip 4 emulator frames), 2472 steps:
    #
    #   |d x_ram|  0: 58.2%   1: 39.5%   2: 2.2%   3: 0.1%
    #   |d y_px|   0: 58.1%   2:  8.5%   4: 32.6%  6: 0.5%   8: 0.2%
    #   by pose:   walk (0,1,4,5) dy in {0,2,4};  walk (2,3) {0};  ladder (8) {0,4}
    #              jump 9/10, fall 11, rope 14 ({0,8}), spring 16 ({0,4,6}) -- all
    #              AIRBORNE, therefore never eligible for detection
    #
    # x: walking advances x_ram ONE unit at a time (a walk step is 4 px, confirmed
    #    by holding RIGHT and logging per emulator frame), so the agent is sampled
    #    at every x_ram value and cannot skip a target. tol_x 0 suffices for a rest
    #    point; tol_x 1 (+-4 px, one movement step) is cheap insurance. tol_x 2 is
    #    twice what is needed.
    # y: y is always EVEN and a climb advances 4 px per gym step, so the sampled
    #    lattice can be offset from the target by 2 (the L4 climb we traced ran
    #    94,90,86,82,78 and hit y78 exactly; a phase-shifted climb would sample
    #    80 then 76 and miss it). tol_y 2 catches either phase and also covers the
    #    largest DETECTABLE dy, which is 4. tol_y 1 would fail on an offset lattice.
    #
    # => the defensible choice is PER-AXIS: tol_x = 1, tol_y = 2. Splitting this
    # into two fields is a behaviour change for every level that uses waypoints, so
    # it is recorded here rather than done silently.
    #
    # Frame skip only bites for points passed AT SPEED: rope carry moves y 8 px per
    # gym step, falls and the spring 6, all larger than a +-2 px window. Those poses
    # are airborne and thus never detected, so today it is a non-issue -- but any
    # future waypoint on a moving/carried segment (or adding a pose to SEED_POSES,
    # as L3 did with the escalator ride 13) reintroduces it.
    waypoint_tolerance: int = 2
    # Apply the reach gate to WAYPOINT pools as well as the progress rungs, so a
    # waypoint is only used as a start once the agent reaches it from reset at
    # least ``reach_threshold`` of the time. Off by default = the historical
    # asymmetry (rungs gated, waypoints not), which is what bootstrapped L2.
    # Experimental: on L3, ungated drilling of unreachable states produced skill
    # that did not compose to reset (0.03%), so gating may spend the budget
    # better -- at the risk of blocking the kind of breakthrough L2 needed.
    #
    # Cold start: the reach EMA is run-local (never inherited), so on a warm
    # start EVERY waypoint is gated out until reset episodes refill it. That
    # cannot deadlock -- the reset rung is always a start candidate -- and at
    # alpha 0.02 a reliably-reached waypoint clears 0.15 in ~8 reset episodes.
    gate_waypoints: bool = False
    # Gate a waypoint on its PREDECESSOR's reach instead of its own. Requires
    # ``gate_waypoints``. Default False = the own-reach rule.
    #
    # WHY. Gating on a waypoint's OWN reach is self-locking at the frontier: the
    # frontier is by definition the point the agent does not yet reach, so its reach
    # is ~0, so it is never sampled, so the skill is never practised, so its reach
    # stays ~0. Measured on L4 (`l4_anchors_v2_1200k`): `Lclimb3_top`, `Low1` and
    # `Low2_launch` each held 100 usable seeds and were sampled 0 times out of 7140
    # episodes, while `Step` -- the point immediately before `Lclimb3_top` -- was
    # reached 0.81 of the time from reset. The agent arrives at the ladder-3 base
    # reliably and dies on the single frame it tops out (an enemy crosses the head;
    # scripted sweep: departures 11-28 frames into the cycle survive, 0-10 die).
    # That is not an unreachable state, it is the frontier, and it was the only
    # thing standing between the run and rung 11.
    #
    # This keeps the protection the own-reach rule was added for. A waypoint whose
    # PREDECESSOR is also unreached stays gated, so the frontier advances exactly
    # one rung at a time and we never drill states the agent genuinely cannot get
    # near -- which is what wasted ~40% of L3's episodes. The first point on the
    # route has no predecessor and is treated as reachable (reset reaches it).
    gate_waypoints_by_predecessor: bool = False
    # Geometry of the waypoint reach test: "box" (ships today) or "sprite".
    #
    # "box" asks whether the agent's POSITION falls inside a tolerance box around the
    # anchor. "sprite" asks the inverse -- whether the agent's 14x18 SPRITE contains the
    # anchor POINT -- which removes the tolerance as a free parameter. In x the two are
    # nearly identical (sprite is one x_ram unit tighter each side); the real difference
    # is y, where the box compares against the anchor with the same small tolerance (tol
    # 2 = +-2 PIXELS) while overlap accepts anywhere in y..y+17.
    #
    # WHY. Measured on L4 floor 7: the agent lands at px 108, is grounded for TWO
    # FRAMES, then is airborne to the ladder at px 144. A 4 px y-window plus a grounded
    # allowlist can only fire in those two frames, so an anchor 4 px off detects nothing
    # -- anchor 28 scored 1.00 against one policy and 0.00 against another. Under sprite
    # overlap with a non-allowlist pose gate both anchors score ~1.00 on both policies.
    #
    # This value is applied to the curriculum's detection AND forced into the reward's
    # params by the trainer, because the historical bug in this area is precisely the
    # two consumers disagreeing about whether a waypoint was reached.
    #
    # Offline pre-flight (debug/l4_detector_sweep.py, 2 policies, 20 episodes, every
    # route point): no waypoint loses a genuine detection. The only points that fire
    # LESS are the four `_launch` pads, where the shipped tol-6 box was firing at the
    # PREDECESSOR'S SEED 16-24 px away, before the agent moved -- a known defect (three
    # other launch pads were deleted for being "a second, wider, misplaced box for the
    # same traversal"). Switching to sprite mode therefore also silently fixes four
    # milestones that currently mark early; that is a reward change riding along with a
    # detection change, so it is called out rather than discovered later.
    #
    # DEFAULT FLIPPED TO "sprite" (2026-09-11). The tolerance is a free parameter with
    # no physically correct value, and the L4 box audit showed both ends of that bind at
    # once: 12 pairs of waypoint boxes OVERLAP where the agent can stand (up to 33 px --
    # the whole Hi chain, plus `Lhi_down_bot`+`Low2` and `Lclimb1_top`+`Rope1_launch`),
    # so one grounded frame credits two waypoints; while a box narrow enough not to
    # overlap misses genuine arrivals (anchor 28 on `Rope1` scored 0.00 against a policy
    # landing 4 px off). Under sprite overlap the same audit gives ZERO standable
    # overlaps, because the acceptance region is the sprite's own 14x18 rather than
    # 49x13.
    #
    # This also switches CAPTURE, not just detection. The flag used to move detection
    # only, with capture pinned to the box so "seeds stay pinned where they are" -- so
    # the 3-seed A/B of this flag could not produce a pool change and did not: the
    # defect it targeted was structurally out of its reach.
    waypoint_reach_mode: str = "sprite"
    # Waypoints to EXCLUDE from seeding: no states captured there, no episodes started
    # there. Detection and any reward term are untouched -- this is only about whether a
    # spot is used as practice material.
    #
    # WHY THIS IS A CONFIG AND NOT A MAP EDIT. `Target.seedable` lives in the map, so
    # turning it off there changes every run at once and leaves no control arm that a
    # config alone can reproduce. That is the mistake `mark_airborne` and
    # `pay_on_target_change` were both added to avoid. Default empty = no change.
    #
    # THE CASE THAT MOTIVATED IT (L4 `Lclimb3_top`, floor 11, measured 2026-09-18).
    # Floor 11 is exposed -- no ceiling, on the kangaroo path -- and holding NOOP there
    # dies in 3-13 frames, always. `admit_requires_survival` therefore admits only the
    # arrivals that happened to land in a benign hazard phase: pool seeds survive a
    # median of 8 NOOP frames while the policy's own arrivals from reset survive 3. So
    # the pool is 2.7x easier than reality AND starting episodes there hands the agent a
    # survivable phase for free, which is precisely the decision it needs to learn. The
    # approach to the floor-12 jump needs 8 steps, so a 3-frame arrival is doomed
    # whatever it does; v6/v13, which crossed, arrive with 12.
    #
    # Excluding such a waypoint pushes practice back to the previous safe spot (here
    # `Step`, same px one floor down, off the patrol route) so the timed climb happens
    # INSIDE the episode. See experiments/003-yeti/level4_notes.md, v16c section.
    seed_waypoint_skip: List[str] = field(default_factory=list)
    # Partition episode starts as reset | MANDATORY | OTHER instead of
    # reset | rungs | one waypoint group. A rung pool is not a distinct kind of
    # start -- "N mandatory targets done" is "standing at waypoint X" -- and it is
    # only ever filled on a fruit pickup, so on a one-fruit level exactly one rung
    # pool exists and took 34.6% of L4 v2's starts while the frontier waypoint got
    # 1.8%. Grouping rungs with the mandatory waypoints removes that duplicate.
    # On L1/L2 mandatory targets ARE the fruits, so that bucket is exactly the old
    # rung set. Default False keeps L1/L2/L3 unchanged.
    split_mandatory_starts: bool = False
    # Score an episode by what it EARNED over what it had left to do, instead of
    # the absolute depth it reached. Absolute depth cannot equalise and rewards
    # inheritance: a deep seed banks most of the goals for free, so it always
    # scores high and 1 - goal_score gives it the smallest sampling weight.
    # Measured on L4 v2: Step (the stuck frontier) scored 0.643, the LOWEST weight
    # of any eligible start, while the already-solved Lfruit_top scored 0.549.
    earned_progress_score: bool = False
    # Keep a captured snapshot only if the agent SURVIVED from it, dropping the
    # `reached_next` shortcut. That shortcut is computed over the WHOLE episode,
    # so a snapshot taken after the episode's deepest point inherits credit for
    # progress made BEFORE it -- which admitted corpses on L4 v3: 5/20 sampled
    # Lclimb3_top states had the death flag already set in the saved bytes and
    # 20/20 died within 60 NOOP steps (Low1_launch 3/20 and 20/20). Pools up to
    # and including Step were clean. Default False because it also changes L1-L3
    # admission, where the leniency was introduced on purpose to protect rare
    # reaches at sparse rungs; needs a 3-seed sweep with a baseline arm first.
    # See _admit_by_play for the alternative (anchor reached_next to save time).
    admit_requires_survival: bool = False
    # Also require the capture to END ITS SURVIVAL WINDOW in a seedable pose (a surface
    # pose, or the L3 escalator ride) rather than merely to be alive. Requires
    # ``admit_requires_survival`` to be meaningful. Default False.
    #
    # WHY. "Alive after N steps" cannot see a state that falls off a platform and is
    # CAUGHT by something. Measured on L4: 81 of 100 `Low2_launch` seeds sit on px 184
    # --
    # floor 12's tile edge, where the agent reads grounded for one frame then falls (0/8
    # survive a NOOP hold; px 188 survives 8/8) -- and the trampoline below keeps them
    # alive a MEDIAN OF 83 STEPS against `min_survival_steps` 30. So 100/100 doomed
    # seeds
    # were admitted and the pool meant to teach the rope-2 crossing taught the
    # fall-bounce
    # loop instead. Raising the threshold is not the fix: it would only have to beat one
    # particular bounce cycle.
    #
    # BLAST RADIUS, measured over existing pools (debug/l4_survival_gate_blast.py): L4
    # rejects 23 of 415 seeds (6%, ALL in `Low2_launch`); L3 rejects 5 of 463 (1%) and
    # keeps 25/25 of `Lesc_top`. An earlier draft of the criterion used SURFACE_POSES
    # and
    # required y to be unchanged, which rejected all 25 escalator rides -- pose 13 is a
    # legitimate seed state and the y test fails an escalator by construction.
    admit_requires_grounded: bool = False
    # RANDOM NO-OP START: draw 0..N extra no-op gym steps after the start state is
    # loaded, to decorrelate arrival phase from the route. 0 = off (current
    # behaviour). The profile's own `random_noop_max` cannot serve this: it fires
    # in the startup SEQUENCE, and this trainer resets gym once then load_states
    # every episode, so it is overwritten and never reaches L2/L3/L4.
    #
    # Why: measured on L4 v4's pools, the bonus countdown at capture (a monotonic
    # clock ~ arrival time) is nearly constant -- Lfruit_top spread 2 across 100
    # captures, Step 103 with median 751 vs max 753. The policy replays one
    # open-loop trajectory at fixed timing, always meets the periodic kangaroos at
    # the same phase, and never learns to read them; every pool is phase-poor as a
    # result (all 20 sampled Step seeds need a 16-24 step pause, 3 distinct values).
    noop_start_max: int = 0
    # "reset" = jitter only reset-origin episodes (a seed keeps representing the
    # situation it was captured for); "all" = jitter seeded episodes too
    # (diversifies pools, but changes what a seed means). Untested either way.
    noop_start_scope: str = "reset"


@dataclass(frozen=True)
class SegmentConfig:
    """Settings used only by ``train_segment.py``."""

    checkpoints: str
    segment: int


@dataclass(frozen=True)
class BackwardCurriculumConfig:
    """Settings used only by ``go_explore_phase2.py``."""

    archive: str
    advance_threshold: float = 20.0
    advance_window: int = 100
    frontier_ratio: float = 0.5


@dataclass(frozen=True)
class RunConfig:
    """Fully-resolved training-run configuration.

    Only one of ``curriculum``, ``segment``, ``backward_curriculum`` should
    be populated per run — each script validates the one it needs is
    present (and that the others are absent).
    """

    training: TrainingConfig
    env: EnvConfig
    reward: RewardConfig
    ppo: PPOConfig = field(default_factory=PPOConfig)
    curriculum: Optional[CurriculumConfig] = None
    segment: Optional[SegmentConfig] = None
    backward_curriculum: Optional[BackwardCurriculumConfig] = None

    # -- construction -------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        if yaml is None:
            raise RuntimeError("PyYAML is required to load RunConfig from YAML")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Config file {path!r} must be a YAML mapping")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunConfig":
        kwargs: Dict[str, Any] = {}

        # Backward-compat: `min_survival_frames` was renamed to
        # `min_survival_steps` (it counts gym steps, not emulator frames).
        cur = data.get("curriculum")
        if isinstance(cur, Mapping) and "min_survival_frames" in cur:
            cur = dict(cur)
            cur.setdefault("min_survival_steps", cur.pop("min_survival_frames"))
            data = dict(data)
            data["curriculum"] = cur

        # Required sub-configs
        kwargs["training"] = _build(TrainingConfig, data.get("training"), "training")
        kwargs["env"] = _build(EnvConfig, data.get("env"), "env")
        kwargs["reward"] = _build(RewardConfig, data.get("reward"), "reward")

        # Optional / defaulted sub-configs
        if "ppo" in data:
            kwargs["ppo"] = _build(PPOConfig, data.get("ppo"), "ppo")

        # Script-specific — at most one should be present
        script_sections = [
            ("curriculum", CurriculumConfig),
            ("segment", SegmentConfig),
            ("backward_curriculum", BackwardCurriculumConfig),
        ]
        present = [name for name, _ in script_sections if name in data]
        if len(present) > 1:
            raise ValueError(
                f"Config has multiple script-specific sections: {present}. "
                "Only one of (curriculum, segment, backward_curriculum) is allowed."
            )
        for name, klass in script_sections:
            if name in data:
                kwargs[name] = _build(klass, data.get(name), name)

        # Reject unknown top-level keys.
        known_top_level = {
            "training",
            "env",
            "reward",
            "ppo",
            "curriculum",
            "segment",
            "backward_curriculum",
        }
        unknown = set(data) - known_top_level
        if unknown:
            raise ValueError(
                f"Unknown top-level config keys: {sorted(unknown)}. "
                f"Expected one or more of {sorted(known_top_level)}."
            )

        return cls(**kwargs)

    # -- persistence --------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict representation, omitting ``None`` sub-configs."""
        result = dataclasses.asdict(self)
        for key in ("curriculum", "segment", "backward_curriculum"):
            if result.get(key) is None:
                result.pop(key, None)
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build(klass, payload, section_name: str):
    """Instantiate ``klass`` from a mapping, rejecting unknown keys."""
    if payload is None:
        raise ValueError(f"Config missing required section {section_name!r}")
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Config section {section_name!r} must be a mapping, "
            f"got {type(payload).__name__}"
        )
    known = {f.name for f in fields(klass)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(
            f"Unknown keys in {section_name!r}: {sorted(unknown)}. "
            f"Expected one or more of {sorted(known)}."
        )
    # Tuples need explicit coercion from YAML lists for the type-annotated
    # fields (``resize`` is the only one currently; keep this generic).
    coerced = {}
    for name, value in payload.items():
        annotation = klass.__dataclass_fields__[name].type
        coerced[name] = _coerce(value, annotation)
    return klass(**coerced)


def _coerce(value, annotation):
    """Best-effort coercion for a handful of types we expect from YAML."""
    # YAML represents tuples as lists. Only coerce when the annotation
    # names a tuple; otherwise return as-is and let the dataclass accept
    # whatever YAML loaded.
    anno_str = str(annotation)
    if value is not None and "Tuple" in anno_str and isinstance(value, list):
        return tuple(value)
    return value


__all__ = [
    "BackwardCurriculumConfig",
    "CurriculumConfig",
    "EnvConfig",
    "PPOConfig",
    "RewardConfig",
    "RunConfig",
    "SegmentConfig",
    "TrainingConfig",
]
