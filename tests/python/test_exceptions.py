"""Tests for the retro-ai exception hierarchy."""

import retro_ai


class TestExceptionHierarchy:
    def test_base_is_exception(self):
        assert issubclass(retro_ai.RetroAIError, Exception)

    def test_initialization_error(self):
        assert issubclass(
            retro_ai.InitializationError, retro_ai.RetroAIError
        )

    def test_invalid_action_error(self):
        assert issubclass(
            retro_ai.InvalidActionError, retro_ai.RetroAIError
        )

    def test_state_error(self):
        assert issubclass(retro_ai.StateError, retro_ai.RetroAIError)

    def test_configuration_error(self):
        assert issubclass(
            retro_ai.ConfigurationError, retro_ai.RetroAIError
        )

    def test_catch_all_with_base(self):
        """All specific errors should be catchable via RetroAIError."""
        for cls in (
            retro_ai.InitializationError,
            retro_ai.InvalidActionError,
            retro_ai.StateError,
            retro_ai.ConfigurationError,
        ):
            try:
                raise cls("test")
            except retro_ai.RetroAIError:
                pass  # expected

    def test_message_preserved(self):
        msg = "something went wrong"
        err = retro_ai.StateError(msg)
        assert str(err) == msg
