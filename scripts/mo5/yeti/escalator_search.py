#!/usr/bin/env python3
"""Phase-1 escalator search, 2-unknown model (X, T) from a SINGLE seed.

From the goat platform the agent must
    walk right X   -> launch position (previous attempts jumped from too far
                      LEFT and fell into the gap short of the escalator),
    wait T         -> phase the jump against the moving platform, then
    jump right     -> RJUMP held JH frames, then hold still (NOOP).
Script: RIGHT*X + NOOP*T + RJUMP*JH + NOOP*RIDE.

SUCCESS CRITERION (corrected): riding the escalator DOWN shows as an airborne
pose (11), not a surface pose, so "grounded while descending" never happens --
the only grounded moments are the goat start and the LANDING. So success =
the agent ends up GROUNDED + ALIVE at the escalator-bottom / eland band
(y in [LAND_Y_LO, LAND_Y_HI]) after having descended from goat level. That is
"a platform caught it at the bottom" (the fatal-fall case sails through this
band and dies at y182 instead).

We sweep the two unknowns X and T (plus a small JH set) from ONE fixed seed
(left goat-ladder top). Prints an X-by-T landscape of the deepest ALIVE y so we
can see how close each combo gets, marks landings, and saves landing states.
"""
from __future__ import annotations

import pickle

from retro_ai.games import yeti
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig

X, Y, POSE, DEATH = 11090, 11089, 11092, 11004
SURF = set(yeti.SURFACE_POSES)

GOAT_Y = 94
ELAND_Y = 158
LAND_Y_LO, LAND_Y_HI = 148, 172  # a safe landing at the escalator bottom
DESC_MIN_Y = 120  # must have descended below this (airborne) to count as "went down"

X_RANGE = range(0, 15)  # walk-right steps (launch position)
T_RANGE = range(0, 31)  # wait steps (platform phase; > one full cycle)
JHOLDS = (3, 4, 5, 6)  # jump-hold frames (arc distance)
RIDE = 45  # no-op frames after the jump (enough to complete the descent + land)


def scan_descent(gym_env, ifc, seq, save_state):
    """Run one script; detect a safe escalator landing.

    Returns (landed_state_or_None, deepest_alive_y, died). ``landed_state`` is a
    save-state taken at the first frame the agent is GROUNDED + ALIVE in the
    landing band after having descended (been airborne below DESC_MIN_Y).
    """
    descended = False
    deepest_alive = GOAT_Y
    landed_state = None
    died = False
    for a in seq:
        gym_env.step(a)
        y, p = ifc.read_ram_byte(Y), ifc.read_ram_byte(POSE)
        died = ifc.read_ram_byte(DEATH) == 65
        if died:
            break
        deepest_alive = max(deepest_alive, y)
        grounded = p in SURF
        if not grounded and y >= DESC_MIN_Y:
            descended = True
        if (
            landed_state is None
            and descended
            and grounded
            and LAND_Y_LO <= y <= LAND_Y_HI
        ):
            landed_state = save_state()
    return landed_state, deepest_alive, died


def main():
    cfg = EnvConfig(
        profile="yeti_fruit_level3",
        action_mode="joystick",
        max_steps=400,
        stall_threshold=400,
        resize=(84, 84),
    )
    stack = build_training_env("yeti_fruit_level3", cfg)
    base, gym_env, ifc = stack.base, stack.gym, stack.base._interface
    base.reset(seed=0)
    d = pickle.load(
        open("output/mo5/yeti/training/yeti_curriculum_l3_v4_15m/checkpoints.pkl", "rb")
    )
    seed = d["waypoints"]["Lgoat_a_top"][0][0][2]  # left goat-ladder top
    print(
        f"1 seed (Lgoat_a_top[0]); sweep X=0..{list(X_RANGE)[-1]} "
        f"T=0..{list(T_RANGE)[-1]} JH={JHOLDS}; land band y[{LAND_Y_LO},{LAND_Y_HI}]",
        flush=True,
    )
    RIGHT, RJUMP, NOOP = [0, 1, 0], [0, 1, 1], [0, 0, 0]

    landing_states = []
    landings = []  # (X, T, JH)
    best_depth = 0
    best_depth_params = None
    # landscape[jh][x][t] = deepest alive y (higher = deeper); 'L' if landed
    land = {jh: [[GOAT_Y] * len(T_RANGE) for _ in X_RANGE] for jh in JHOLDS}
    landed_grid = {jh: [[False] * len(T_RANGE) for _ in X_RANGE] for jh in JHOLDS}

    for jh in JHOLDS:
        for xi, x_walk in enumerate(X_RANGE):
            for ti, t_wait in enumerate(T_RANGE):
                ifc.load_state(seed)
                stack.preprocessed.notify_state_loaded()
                for _ in range(5):
                    gym_env.step(NOOP)
                seq = [RIGHT] * x_walk + [NOOP] * t_wait + [RJUMP] * jh + [NOOP] * RIDE
                landed_state, deepest, _died = scan_descent(
                    gym_env, ifc, seq, ifc.save_state
                )
                land[jh][xi][ti] = deepest
                if deepest > best_depth:
                    best_depth = deepest
                    best_depth_params = (x_walk, t_wait, jh)
                if landed_state is not None:
                    landed_grid[jh][xi][ti] = True
                    landings.append((x_walk, t_wait, jh))
                    landing_states.append(landed_state)
        print(f"  JH={jh} done; landings so far={len(landings)}", flush=True)

    total = len(JHOLDS) * len(X_RANGE) * len(T_RANGE)
    print(
        f"\nRESULT: landings={len(landings)} of {total} scripts; "
        f"best_depth_alive={best_depth} (goat={GOAT_Y}, eland={ELAND_Y}, "
        f"bottom=182) params(X,T,JH)={best_depth_params}",
        flush=True,
    )
    # Landscape for the JH of the best depth: deepest-alive y per (X,T),
    # bucketed; 'L' = a safe landing, '#' = died deep (reached bottom & died).
    show_jh = best_depth_params[2] if best_depth_params else JHOLDS[0]
    print(
        f"\nLandscape JH={show_jh} (rows=X, cols=T; digit=(deepest_alive-94)//10, "
        "L=landed):",
        flush=True,
    )
    print("     " + "".join(str(t % 10) for t in T_RANGE), flush=True)
    for xi, x_walk in enumerate(X_RANGE):
        row = ""
        for ti in range(len(T_RANGE)):
            if landed_grid[show_jh][xi][ti]:
                row += "L"
            else:
                row += str(min(9, max(0, (land[show_jh][xi][ti] - GOAT_Y) // 10)))
        print(f"  X{x_walk:02d} {row}", flush=True)

    if landings:
        print(f"\nLandings (X,T,JH): {landings[:30]}", flush=True)
        pickle.dump(
            {"escalator_boards": landing_states},
            open("debug/escalator_boards.pkl", "wb"),
        )
        print(f"saved {len(landing_states)} landing states", flush=True)
    else:
        print("\nNO safe landings found across the (X,T,JH) sweep.", flush=True)


if __name__ == "__main__":
    main()
