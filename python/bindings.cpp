/**
 * @file bindings.cpp
 * @brief pybind11 Python bindings for the retro-ai C++ core.
 *
 * Exposes ObservationSpace, ActionSpace, ActionType, StepResult, and
 * RLInterface to Python via the retro_ai_native module.
 *
 * Key design decisions:
 *   - Zero-copy NumPy array conversion for observations (via py::buffer_protocol)
 *   - GIL release during computationally intensive operations (step, reset)
 *   - C++ exceptions are translated to Python exceptions
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "retro_ai/rl_interface.hpp"
#include "retro_ai/exceptions.hpp"

#ifdef HAVE_VIDEOPAC
#include "retro_ai/videopac_rl.hpp"
#endif

#ifdef HAVE_MO5
#include "retro_ai/mo5_rl.hpp"
#endif

namespace py = pybind11;

using namespace retro_ai;

// ---------------------------------------------------------------------------
// Helper: convert a flat std::vector<uint8_t> observation into a NumPy array
// with shape (height, width, channels).  The returned array owns a copy of the
// data so it remains valid after the C++ side mutates its internal buffer.
// When the observation size matches the expected dimensions we reshape;
// otherwise we return a flat 1-D array so the caller can still inspect it.
// ---------------------------------------------------------------------------
static py::array_t<uint8_t> observation_to_numpy(
    const std::vector<uint8_t>& obs,
    const ObservationSpace& space)
{
    const auto expected = static_cast<size_t>(
        space.width * space.height * space.channels);

    if (obs.size() == expected && expected > 0) {
        // Return a 3-D array (height, width, channels) with C-contiguous strides
        py::array_t<uint8_t> arr({space.height, space.width, space.channels});
        std::memcpy(arr.mutable_data(), obs.data(), obs.size());
        return arr;
    }

    // Fallback: return flat copy
    py::array_t<uint8_t> arr(static_cast<py::ssize_t>(obs.size()));
    std::memcpy(arr.mutable_data(), obs.data(), obs.size());
    return arr;
}

// ---------------------------------------------------------------------------
// Helper: wrap a StepResult so that the observation is a NumPy array.
// We return a py::dict so Python code can access .observation as ndarray.
// ---------------------------------------------------------------------------
static py::dict step_result_to_dict(
    const StepResult& result,
    const ObservationSpace& space)
{
    py::dict d;
    d["observation"] = observation_to_numpy(result.observation, space);
    d["reward"]      = result.reward;
    d["done"]        = result.done;
    d["truncated"]   = result.truncated;
    d["info"]        = result.info;
    return d;
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(retro_ai_native, m) {
    m.doc() = "retro-ai native C++ bindings";

    // -- Exception hierarchy ------------------------------------------------
    // Register C++ exceptions so they map to catchable Python exceptions.
    // The hierarchy mirrors the C++ side: RetroAIException is the base,
    // with InitializationError, InvalidActionError, and StateError derived.
    static auto py_retro_ai_error = py::register_exception<RetroAIException>(
        m, "RetroAIError");
    py::register_exception<InitializationError>(
        m, "InitializationError", py_retro_ai_error);
    py::register_exception<InvalidActionError>(
        m, "InvalidActionError", py_retro_ai_error);
    py::register_exception<StateError>(
        m, "StateError", py_retro_ai_error);

    // -- ActionType enum ----------------------------------------------------
    py::enum_<ActionType>(m, "ActionType")
        .value("DISCRETE",       ActionType::DISCRETE)
        .value("MULTI_DISCRETE", ActionType::MULTI_DISCRETE)
        .value("CONTINUOUS",     ActionType::CONTINUOUS)
        .export_values();

    // -- ObservationSpace ---------------------------------------------------
    py::class_<ObservationSpace>(m, "ObservationSpace")
        .def(py::init<>())
        .def_readwrite("width",           &ObservationSpace::width)
        .def_readwrite("height",          &ObservationSpace::height)
        .def_readwrite("channels",        &ObservationSpace::channels)
        .def_readwrite("bits_per_channel", &ObservationSpace::bits_per_channel)
        .def("__repr__", [](const ObservationSpace& s) {
            return "<ObservationSpace " +
                   std::to_string(s.width) + "x" +
                   std::to_string(s.height) + "x" +
                   std::to_string(s.channels) + " @ " +
                   std::to_string(s.bits_per_channel) + "bpc>";
        });

    // -- ActionSpace --------------------------------------------------------
    py::class_<ActionSpace>(m, "ActionSpace")
        .def(py::init<>())
        .def_readwrite("type",  &ActionSpace::type)
        .def_readwrite("shape", &ActionSpace::shape)
        .def("__repr__", [](const ActionSpace& s) {
            std::string shape_str = "[";
            for (size_t i = 0; i < s.shape.size(); ++i) {
                if (i > 0) shape_str += ", ";
                shape_str += std::to_string(s.shape[i]);
            }
            shape_str += "]";
            return "<ActionSpace type=" +
                   std::to_string(static_cast<int>(s.type)) +
                   " shape=" + shape_str + ">";
        });

    // -- StepResult ---------------------------------------------------------
    py::class_<StepResult>(m, "StepResult")
        .def(py::init<>())
        .def_readwrite("observation", &StepResult::observation)
        .def_readwrite("reward",      &StepResult::reward)
        .def_readwrite("done",        &StepResult::done)
        .def_readwrite("truncated",   &StepResult::truncated)
        .def_readwrite("info",        &StepResult::info)
        .def("observation_numpy", [](const StepResult& self,
                                     const ObservationSpace& space) {
            return observation_to_numpy(self.observation, space);
        }, py::arg("space"),
           "Return the observation as a NumPy array shaped (H, W, C).");

    // -- RLInterface (abstract) ---------------------------------------------
    // We use a trampoline class so Python can (optionally) subclass it,
    // but the primary use-case is consuming concrete C++ implementations.
    class PyRLInterface : public RLInterface {
    public:
        using RLInterface::RLInterface;

        StepResult reset(int seed = -1) override {
            PYBIND11_OVERRIDE_PURE(StepResult, RLInterface, reset, seed);
        }
        StepResult step(const std::vector<int>& action) override {
            PYBIND11_OVERRIDE_PURE(StepResult, RLInterface, step, action);
        }
        ObservationSpace observation_space() const override {
            PYBIND11_OVERRIDE_PURE(ObservationSpace, RLInterface, observation_space);
        }
        ActionSpace action_space() const override {
            PYBIND11_OVERRIDE_PURE(ActionSpace, RLInterface, action_space);
        }
        std::vector<uint8_t> save_state() const override {
            PYBIND11_OVERRIDE_PURE(std::vector<uint8_t>, RLInterface, save_state);
        }
        void load_state(const std::vector<uint8_t>& state) override {
            PYBIND11_OVERRIDE_PURE(void, RLInterface, load_state, state);
        }
        void set_reward_mode(const std::string& mode) override {
            PYBIND11_OVERRIDE_PURE(void, RLInterface, set_reward_mode, mode);
        }
        std::vector<std::string> available_reward_modes() const override {
            PYBIND11_OVERRIDE_PURE(std::vector<std::string>, RLInterface, available_reward_modes);
        }
        std::string emulator_name() const override {
            PYBIND11_OVERRIDE_PURE(std::string, RLInterface, emulator_name);
        }
        std::string game_name() const override {
            PYBIND11_OVERRIDE_PURE(std::string, RLInterface, game_name);
        }
    };

    py::class_<RLInterface, PyRLInterface, std::shared_ptr<RLInterface>>(m, "RLInterface")
        // reset() – release the GIL during emulation work
        .def("reset", [](RLInterface& self, int seed) {
            StepResult result;
            {
                py::gil_scoped_release release;
                result = self.reset(seed);
            }
            return result;
        }, py::arg("seed") = -1,
           "Reset the environment and return the initial StepResult.")

        // step() – release the GIL during emulation work
        .def("step", [](RLInterface& self, const std::vector<int>& action) {
            StepResult result;
            {
                py::gil_scoped_release release;
                result = self.step(action);
            }
            return result;
        }, py::arg("action"),
           "Execute one step and return a StepResult.")

        // Convenience wrappers that return NumPy observations directly
        .def("reset_numpy", [](RLInterface& self, int seed) {
            StepResult result;
            {
                py::gil_scoped_release release;
                result = self.reset(seed);
            }
            return step_result_to_dict(result, self.observation_space());
        }, py::arg("seed") = -1,
           "Reset and return a dict with observation as NumPy array.")

        .def("step_numpy", [](RLInterface& self, const std::vector<int>& action) {
            StepResult result;
            {
                py::gil_scoped_release release;
                result = self.step(action);
            }
            return step_result_to_dict(result, self.observation_space());
        }, py::arg("action"),
           "Step and return a dict with observation as NumPy array.")

        // Query methods (no GIL release needed – fast, read-only)
        .def("observation_space", &RLInterface::observation_space)
        .def("action_space",      &RLInterface::action_space)

        // State management – release GIL for potentially heavy I/O
        .def("save_state", [](const RLInterface& self) {
            std::vector<uint8_t> state;
            {
                py::gil_scoped_release release;
                state = self.save_state();
            }
            return py::bytes(reinterpret_cast<const char*>(state.data()),
                             state.size());
        }, "Save the current emulator state as bytes.")

        .def("load_state", [](RLInterface& self, const py::bytes& data) {
            std::string raw = data;
            std::vector<uint8_t> state(raw.begin(), raw.end());
            {
                py::gil_scoped_release release;
                self.load_state(state);
            }
        }, py::arg("state"),
           "Load a previously saved emulator state.")

        // Reward configuration
        .def("set_reward_mode",      &RLInterface::set_reward_mode,
             py::arg("mode"),
             "Set the reward computation mode.")
        .def("available_reward_modes", &RLInterface::available_reward_modes,
             "Return a list of available reward mode names.")

        // Metadata
        .def("emulator_name", &RLInterface::emulator_name)
        .def("game_name",     &RLInterface::game_name);

    // -- Helper: observation_to_numpy standalone function --------------------
    m.def("observation_to_numpy",
          &observation_to_numpy,
          py::arg("observation"),
          py::arg("space"),
          "Convert a flat uint8 observation vector to a shaped NumPy array.");

    // -- Concrete emulator adapters (conditionally compiled) ----------------
#ifdef HAVE_VIDEOPAC
    py::class_<VideopacRLInterface, RLInterface, std::shared_ptr<VideopacRLInterface>>(
            m, "VideopacRLInterface")
        .def(py::init<const std::string&, const std::string&, const std::string&, int>(),
             py::arg("bios_path"),
             py::arg("rom_path"),
             py::arg("reward_mode") = "survival",
             py::arg("joystick_index") = 0,
             "Create a Videopac (Odyssey 2) environment.");
#endif

#ifdef HAVE_MO5
    py::class_<MO5RLInterface, RLInterface, std::shared_ptr<MO5RLInterface>>(
            m, "MO5RLInterface")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("rom_path"),
             py::arg("reward_mode") = "survival",
             "Create a Thomson MO5 environment.");
#endif
}
