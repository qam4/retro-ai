"""Smoke tests for tasks 2.1-2.5: game profile system."""

import os
import json
import tempfile

import pytest

from retro_ai.training.game_profile import (
    StartupAction,
    StartupSequence,
    GameProfile,
    GameProfileRegistry,
)
from retro_ai.training.config import (
    TrainingConfig,
    merge_config_with_profile,
)


# ---- Task 2.1: Dataclasses ------------------------------------------------

class TestStartupAction:
    def test_defaults(self):
        sa = StartupAction(action=1)
        assert sa.action == 1
        assert sa.frames == 1

    def test_custom_frames(self):
        sa = StartupAction(action=3, frames=10)
        assert sa.frames == 10


class TestStartupSequence:
    def test_defaults(self):
        ss = StartupSequence()
        assert ss.actions == []
        assert ss.post_delay_frames == 60

    def test_with_actions(self):
        ss = StartupSequence(
            actions=[StartupAction(action=1, frames=5)],
            post_delay_frames=120,
        )
        assert len(ss.actions) == 1
        assert ss.post_delay_frames == 120


class TestGameProfile:
    def test_required_fields(self):
        gp = GameProfile(
            name="test", emulator_type="videopac", rom_path="/rom.bin"
        )
        assert gp.name == "test"
        assert gp.grayscale is True
        assert gp.resize == (84, 84)
        assert gp.frame_stack == 4
        assert gp.frame_skip == 4
        assert gp.startup_sequence is None
        assert gp.bios_path is None
        assert gp.display_name == ""
        assert gp.action_count is None

    def test_from_yaml(self):
        path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..",
            "game_profiles",
            "videopac_satellite_attack.yaml",
        )
        profile = GameProfile.from_yaml(path)
        assert profile.name == "satellite_attack"
        assert profile.emulator_type == "videopac"
        assert profile.startup_sequence is not None
        assert len(profile.startup_sequence.actions) == 1
        assert profile.startup_sequence.actions[0].action == 1
        assert profile.startup_sequence.actions[0].frames == 10
        assert profile.startup_sequence.post_delay_frames == 120
        assert profile.resize == (84, 84)

    def test_from_json(self):
        gp = GameProfile(
            name="test_json",
            emulator_type="mo5",
            rom_path="/test.rom",
            startup_sequence=StartupSequence(
                actions=[StartupAction(action=2, frames=5)],
                post_delay_frames=30,
            ),
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(gp.to_dict(), f)
            tmp = f.name
        try:
            loaded = GameProfile.from_json(tmp)
            assert loaded.name == "test_json"
            assert loaded.startup_sequence is not None
            assert loaded.startup_sequence.actions[0].action == 2
            assert loaded.resize == (84, 84)
        finally:
            os.unlink(tmp)

    def test_round_trip(self):
        gp = GameProfile(
            name="rt",
            emulator_type="videopac",
            rom_path="/rom.bin",
            startup_sequence=StartupSequence(
                actions=[StartupAction(action=1, frames=10)],
                post_delay_frames=120,
            ),
        )
        d = gp.to_dict()
        gp2 = GameProfile._deserialize(d)
        assert gp2.name == gp.name
        assert gp2.emulator_type == gp.emulator_type
        assert gp2.resize == gp.resize
        assert gp2.startup_sequence.post_delay_frames == 120


# ---- Task 2.2: Registry ---------------------------------------------------

class TestGameProfileRegistry:
    def test_list_profiles(self):
        profiles_dir = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..",
            "game_profiles",
        )
        registry = GameProfileRegistry(profile_dirs=[profiles_dir])
        names = registry.list_profiles()
        assert "satellite_attack" in names

    def test_load_by_name(self):
        profiles_dir = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..",
            "game_profiles",
        )
        registry = GameProfileRegistry(profile_dirs=[profiles_dir])
        profile = registry.load("satellite_attack")
        assert profile.name == "satellite_attack"

    def test_load_by_path(self):
        path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..",
            "game_profiles",
            "videopac_satellite_attack.yaml",
        )
        registry = GameProfileRegistry()
        profile = registry.load(path)
        assert profile.name == "satellite_attack"

    def test_load_missing_raises(self):
        registry = GameProfileRegistry(profile_dirs=[])
        with pytest.raises(Exception):
            registry.load("nonexistent_game")


# ---- Task 2.3: Config merge -----------------------------------------------

class TestMergeConfigWithProfile:
    def test_profile_fills_none_fields(self):
        tc = TrainingConfig()
        gp = GameProfile(
            name="test",
            emulator_type="videopac",
            rom_path="/rom.bin",
            bios_path="/bios.bin",
        )
        merged = merge_config_with_profile(tc, gp)
        assert merged.emulator_type == "videopac"
        assert merged.rom_path == "/rom.bin"
        assert merged.bios_path == "/bios.bin"

    def test_explicit_tc_wins(self):
        tc = TrainingConfig(emulator_type="mo5", rom_path="/my.rom")
        gp = GameProfile(
            name="test",
            emulator_type="videopac",
            rom_path="/other.rom",
        )
        merged = merge_config_with_profile(tc, gp)
        assert merged.emulator_type == "mo5"
        assert merged.rom_path == "/my.rom"

    def test_non_default_tc_wins_over_profile(self):
        tc = TrainingConfig(frame_stack=8)
        gp = GameProfile(
            name="test",
            emulator_type="videopac",
            rom_path="/rom.bin",
            frame_stack=2,
        )
        merged = merge_config_with_profile(tc, gp)
        assert merged.frame_stack == 8

    def test_profile_overrides_default(self):
        tc = TrainingConfig()  # frame_skip=4 (default)
        gp = GameProfile(
            name="test",
            emulator_type="videopac",
            rom_path="/rom.bin",
            frame_skip=2,
        )
        merged = merge_config_with_profile(tc, gp)
        assert merged.frame_skip == 2
