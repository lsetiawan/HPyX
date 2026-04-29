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

#include <cstdint>
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

// Build a PolicyToken from ints inside C++. Called from the binding functions.
static hpyx::policy::PolicyToken make_token(
    int kind, bool task, int chunk, std::size_t chunk_size)
{
    return hpyx::policy::PolicyToken{
        static_cast<hpyx::policy::Kind>(kind),
        task,
        static_cast<hpyx::policy::ChunkKind>(chunk),
        chunk_size,
    };
}

// ---- for_loop (synchronous) ----

static void parallel_for_loop(
    int kind, bool task, int chunk, std::size_t chunk_size,
    std::int64_t first,
    std::int64_t last,
    nb::callable body)
{
    ensure_runtime();
    auto tok = make_token(kind, task, chunk, chunk_size);

    auto pyfn = [body](std::int64_t i) {
        HPYX_CALLBACK_GIL;
        body(i);
    };

    nb::gil_scoped_release release;
    hpyx::policy::dispatch_policy(tok, [&](auto&& policy) {
        hpx::experimental::for_loop(policy, first, last, pyfn);
    });
}

// ---- for_each (synchronous) ----

static void parallel_for_each(
    int kind, bool task, int chunk, std::size_t chunk_size,
    nb::iterable iterable,
    nb::callable body)
{
    ensure_runtime();
    auto tok = make_token(kind, task, chunk, chunk_size);

    std::vector<nb::object> items;
    for (auto item : iterable) {
        items.push_back(nb::borrow(item));
    }

    auto pyfn = [body](nb::object& item) {
        HPYX_CALLBACK_GIL;
        body(item);
    };

    nb::gil_scoped_release release;
    hpyx::policy::dispatch_policy(tok, [&](auto&& policy) {
        hpx::for_each(policy, items.begin(), items.end(), pyfn);
    });
}

void register_bindings(nb::module_& m) {
    m.def("for_loop", &parallel_for_loop,
          "kind"_a, "task"_a, "chunk"_a, "chunk_size"_a,
          "first"_a, "last"_a, "body"_a,
          "Invoke body(i) for i in [first, last) under the given execution policy.");
    m.def("for_each", &parallel_for_each,
          "kind"_a, "task"_a, "chunk"_a, "chunk_size"_a,
          "iterable"_a, "body"_a,
          "Apply body(x) to every element in iterable under the given execution policy.");
}

}  // namespace hpyx::parallel

