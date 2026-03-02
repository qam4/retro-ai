#pragma once

#include <stdexcept>
#include <string>

namespace retro_ai {

/// Base exception for all retro-ai framework errors.
class RetroAIException : public std::runtime_error {
public:
    explicit RetroAIException(const std::string& message)
        : std::runtime_error(message) {}
};

/// Raised when environment initialization fails (e.g. invalid ROM/BIOS paths).
class InitializationError : public RetroAIException {
public:
    explicit InitializationError(const std::string& message)
        : RetroAIException("Initialization failed: " + message) {}
};

/// Raised when an out-of-range or otherwise invalid action is provided.
class InvalidActionError : public RetroAIException {
public:
    explicit InvalidActionError(int action, int max_action)
        : RetroAIException("Invalid action " + std::to_string(action) +
                           ", must be in range [0, " + std::to_string(max_action) + ")") {}
};

/// Raised when a state save/load operation fails.
class StateError : public RetroAIException {
public:
    explicit StateError(const std::string& message)
        : RetroAIException("State operation failed: " + message) {}
};

}  // namespace retro_ai
