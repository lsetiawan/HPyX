#include "parallel.hpp"
#include "gil_macros.hpp"
#include "policy_dispatch.hpp"
#include "futures.hpp"
#include "runtime.hpp"

#include <hpx/algorithm.hpp>
#include <hpx/parallel/algorithms/for_loop.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace hpyx::parallel {

namespace {

void ensure_runtime() {
    if (!hpyx::runtime::runtime_is_running()) {
        throw std::runtime_error(
            "HPyX runtime is not running. Call hpyx.init() first.");
    }
}

}  // namespace

// Sequential for_loop — called from the main Python thread with GIL held.
// The Python wrapper dispatches par/par_unseq policies via hpyx.async_
// to avoid free-threaded Python thread-state races in HPX's work-stealing.
static void parallel_for_loop(
    int kind, bool task_flag, int chunk, std::size_t chunk_size,
    std::int64_t first,
    std::int64_t last,
    nb::callable body)
{
    ensure_runtime();
    for (std::int64_t i = first; i < last; ++i) {
        body(nb::int_(i));
    }
}

// Sequential for_each — iterates and calls fn(item) for each element.
static void parallel_for_each(
    int kind, bool task_flag, int chunk, std::size_t chunk_size,
    nb::iterable iterable,
    nb::callable body)
{
    ensure_runtime();
    for (auto item : iterable) {
        body(item);
    }
}

void register_bindings(nb::module_& m) {
    m.def("for_loop", &parallel_for_loop,
          "kind"_a, "task"_a, "chunk"_a, "chunk_size"_a,
          "first"_a, "last"_a, "body"_a);
    m.def("for_each", &parallel_for_each,
          "kind"_a, "task"_a, "chunk"_a, "chunk_size"_a,
          "iterable"_a, "body"_a);
}

}  // namespace hpyx::parallel
