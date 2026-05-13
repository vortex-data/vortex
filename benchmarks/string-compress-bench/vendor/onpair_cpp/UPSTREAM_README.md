# OnPair

[![CI](https://github.com/gargiulofrancesco/onpair_cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/gargiulofrancesco/onpair_cpp/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/gargiulofrancesco/onpair_cpp/branch/main/graph/badge.svg)](https://codecov.io/gh/gargiulofrancesco/onpair_cpp)
![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C.svg)
![CMake](https://img.shields.io/badge/CMake-3.21%2B-064F8C.svg)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Field-level string compression for database systems, with random access and compressed-domain string predicates.**

> **Status:** OnPair is pre-1.0. The on-disk format and the public API may change without notice. There are no tagged releases yet — pin to a specific commit hash if you embed the library.

OnPair stores a string column as a bit-packed token stream backed by a learned, size-configurable dictionary of frequent patterns (9–16 bits per token, 512–65,536 entries). It is built for execution engines that need to decompress individual values by row id, and accelerates the evaluation of SQL-style predicates by running them directly on the compressed stream.

## Overview

General-purpose compressors (LZ4, Zstd, Snappy) group many short strings into a shared block to reach a useful ratio. That block is a decoding dependency: reading one row, or scanning for a pattern, forces the engine to decompress every byte of the block — successive strings cannot be reconstructed without the ones before them.

OnPair encodes each string independently. Random access decodes only the bytes for the requested row id, and a pattern scan (e.g., `LIKE '%needle%'`) can stop on the first match in a row and jump to the next row without touching the rest. A column-level dictionary amortises the cost of frequent byte patterns across the whole column, and search predicates are compiled into token automata and evaluated directly over the compressed stream:

- Substring match: `LIKE '%needle%'` 
- Prefix match: `LIKE 'needle%'`
- Equality: `WHERE col = 'value'`
- Multi-pattern match:  `LIKE '%a%' OR LIKE '%b%' OR …`
- Boolean composition: `NOT`, `AND`, `OR` over any of the above

The same physical representation supports point decompression, full-column scans, and the predicate set above — without secondary indexes or operator-specific layouts.

## Implementation Properties

- **Compressed-domain boolean algebra.** Arbitrary boolean compositions of substring, prefix, equality, and multi-pattern predicates execute in one pass over the compressed stream — no separate filter, no intermediate materialisation.
- **Sorted dictionary.** Tokens are stored lexicographically, enabling binary-search prefix ranges and sparse automaton transition ranges.
- **Amortised query compilation.** Each predicate compiles once against the column's dictionary, then scans an arbitrary number of rows without re-tokenising or rebuilding transitions.
- **Bit-packed fixed-width store.** Token ids are packed LSB-first at 9–16 bits per token with Arrow-style `n + 1` row boundaries.
- **Compile-time bit-width dispatch.** Runtime bit width is resolved once at column open and specialises every hot loop, so 9–16-bit columns share no shifts or masks at run time.
- **Range-based and Arrow-compatible API.** Compresses any C++20 range of `std::string_view`-convertible values, and accepts Arrow-style `(bytes, offsets, n)` buffers directly.
- **Versioned binary persistence.** Columns serialize to `ONPAIR01` plus dictionary and packed-store arrays.


## Quick Start

```cpp
#include <onpair/api.h>

#include <cstddef>
#include <string_view>
#include <vector>

namespace op = onpair;

int main() {
    // 1. Compress any C++20 range of string-like values.
    std::vector<std::string_view> data = {
        "user_000001", "user_000002", "admin_001",
        "user_000003", "guest_001",   "admin_002",
    };

    op::encoding::TrainingConfig cfg;
    cfg.bits      = 14;                                // 16,384-token dictionary
    cfg.threshold = op::encoding::DynamicThreshold{0.15};
    cfg.seed      = 42;                                // reproducible dictionary

    op::OnPairColumn col = op::OnPairColumn::compress(data, cfg);
    op::OnPairColumnView view = col.view();

    // 2. Random access. The buffer needs decoder padding for over-copy.
    std::vector<char> buf(256 + op::DECOMPRESS_BUFFER_PADDING);
    std::size_t len = view.decompress(0, buf.data());  // "user_000001"
    std::string_view value(buf.data(), len);

    // 3. Convenience APIs return row-id vectors.
    auto admin_hits = view.contains("admin");          // LIKE '%admin%'
    auto user_hits  = view.starts_with("user_");       // LIKE 'user_%'
    auto exact_hit  = view.equals("admin_001");        // WHERE col = 'admin_001'

    // 4. Callback APIs avoid allocating a result vector.
    view.contains("admin", [](std::size_t row_id) {
        // Consume matching row ids directly.
    });

    // 5. Multi-pattern search.
    std::vector<std::string_view> patterns = {"admin", "guest"};
    op::search::AhoCorasickAutomaton ac(patterns, view.dictionary());
    view.scan(ac, [](std::size_t row_id) {
        // LIKE '%admin%' OR LIKE '%guest%'
    });

    // 6. Boolean algebra over compressed-domain predicates.
    op::search::KmpAutomaton kmp_user("user", view.dictionary());
    op::search::KmpAutomaton kmp_guest("guest", view.dictionary());

    auto rows = view.scan(kmp_user && !kmp_guest);     // user AND NOT guest
    return 0;
}
```

The scan loop drives automata over token ids. Use callback overloads when the caller already has a selection-vector builder, bitmap writer, or downstream operator sink.

## Advanced Usage

### Automata Combinators

`TokenAutomaton` is the small execution contract used by the scan loop:

```cpp
void step(onpair::Token token);
bool is_accepted() const;
void reset();
```

Automata may also expose `is_dead()` for early exit. Substring and multi-pattern automata become dead once a match is found; prefix and equality automata become dead once the result can no longer change.

Combinators are lightweight reference wrappers:

```cpp
op::search::KmpAutomaton blocked("@spam.com", view.dictionary());
op::search::KmpAutomaton bounced("bounced",   view.dictionary());
op::search::PrefixAutomaton internal("svc_",  view.dictionary());

view.scan(!blocked && (bounced || internal), [](std::size_t row_id) {
    // NOT blocked AND (bounced OR internal)
});
```

Keep component automata alive for the duration of the scan. Pass composed expressions directly to `scan`, or name intermediate wrappers explicitly when storing them.

### Arrow-Style Buffers

Use the raw-buffer overload when the input already exists as a contiguous byte buffer plus offsets:

```cpp
const char*     bytes   = arrow_string_array.data();
const uint32_t* offsets = arrow_string_array.offsets(); // n + 1 entries
std::size_t     n       = arrow_string_array.length();

op::OnPairColumn col = op::OnPairColumn::compress(bytes, offsets, n, cfg);
```

### Serialization

```cpp
std::ofstream out("column.onp", std::ios::binary);
col.write_to(out);

std::ifstream in("column.onp", std::ios::binary);
op::OnPairColumn restored = op::OnPairColumn::read_from(in);
```

## Benchmarking

Benchmark results, reference datasets, and a comparison protocol against general-purpose codecs (LZ4, Zstd, Snappy) will be published here once the harness in `bench/` is finalised.

For build-time guidance on how to compile OnPair and your benchmark targets for peak performance, see [§Build → Building for performance](#building-for-performance).

## Robustness & CI

The test suite is GoogleTest-based and split by module: core storage, dictionary views, encoding, parsing, decoding, automata, search combinators, serialization, and column integration.

CI runs on:

- **Linux GCC 14** in Debug and Release.
- **Linux Clang 18** in Debug and Release.
- **macOS AppleClang** in Debug and Release.
- **Windows MSVC** in Debug and Release.

Additional CI jobs enforce:

- **ASan + UBSan** on Clang 18.
- **TSan** on Clang 18.
- **Codecov upload** from the GCC coverage build on `main`.
- Weekly scheduled runs in addition to push and pull-request validation.

Local test run:

```bash
cmake -B build -DONPAIR_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure --parallel 4
```

Sanitizer run:

```bash
cmake -B build_san \
  -DONPAIR_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"

cmake --build build_san --parallel
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  ctest --test-dir build_san --output-on-failure --parallel 4
```

## Integration

### Requirements

- C++20 compiler: GCC 11+, Clang 13+, AppleClang, or MSVC 19.29+.
- CMake 3.21+.
- Boost.Unordered ≥ 1.81. If unavailable as a system package, CMake fetches Boost through `FetchContent`.

OnPair is built as a STATIC + PIC archive. `BUILD_SHARED_LIBS=ON` is rejected at configure time — most of OnPair's hot path is templated and instantiated in the consumer's translation units, so a shared `libonpair` would ship only a thin shell of the library. The supported deployment is a STATIC archive linked into your final binary or host DSO.

### Build

A standard Release build produces a portable archive — safe to ship, safe to install:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Both LTO and host-native codegen are off by default so that installed archives remain portable across CPUs and link-compatible with downstream consumers.

CMake options that affect the build:

| Option | Default | Effect |
|---|---|---|
| `BUILD_SHARED_LIBS` | — | Rejected at configure time. OnPair is STATIC-only by design (see §Requirements). |
| `ONPAIR_ENABLE_LTO` | `OFF` | Enables IPO/LTO in any optimised configuration. PRIVATE — does not propagate to consumers. |
| `ONPAIR_NATIVE_ARCH` | `OFF` | Adds `-march=native` to OnPair's own targets. PRIVATE. Binary not portable across CPUs. |
| `ONPAIR_BUILD_TESTS` | `OFF` | Build the GoogleTest suite. |
| `ONPAIR_BUILD_EXAMPLES` | `ON` (top-level) / `OFF` (embedded) | Build the programs under `examples/`. |
| `ONPAIR_INSTALL` | `ON` (top-level) / `OFF` (embedded) | Emit install rules and `find_package(OnPair)` config. |

#### Building for performance

For benchmarks, profiling runs, or single-host deployments, enable LTO and host-native codegen. The resulting binaries are **not portable** to other CPUs.

> **Critical for benchmarking.** OnPair's hot path (decoder, scan loop, automata) lives in templated headers and is instantiated in **your** translation units. `-DONPAIR_ENABLE_LTO=ON -DONPAIR_NATIVE_ARCH=ON` tunes OnPair's own `.cpp` files only — your benchmark or application binary needs equivalent flags too, or its codegen will dominate runtime.

**Standalone build:**

```bash
cmake -B build-bench \
    -DCMAKE_BUILD_TYPE=Release \
    -DONPAIR_ENABLE_LTO=ON \
    -DONPAIR_NATIVE_ARCH=ON
cmake --build build-bench --parallel
```

**Consuming OnPair from a parent CMake** (`add_subdirectory`, `FetchContent`, or `find_package`):

```cmake
add_executable(my_benchmark bench.cpp)
target_link_libraries(my_benchmark PRIVATE OnPair::onpair)

# Apply -march=native and LTO to YOUR target.
# Required: header-only hot paths are codegen'd here.
onpair_optimize_target(my_benchmark)
```

`onpair_optimize_target()` is registered as soon as OnPair is added to the build (via any of the three integration modes) and applies `-march=native` and IPO/LTO to the named target. If you prefer raw flags, compile your benchmark with `-O3 -march=native -flto` (or MSVC equivalents). Applying `-march=native` only to the OnPair archive is the single most common cause of misleading benchmark numbers.

### Consuming OnPair

`OnPair::onpair` is the canonical alias and works identically whether OnPair is consumed via `find_package()`, `FetchContent`, or `add_subdirectory()`.

#### find_package

Install OnPair to a chosen prefix:

```bash
cmake --install build --prefix /opt/onpair
```

Downstream projects then consume it with:

```cmake
find_package(OnPair CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE OnPair::onpair)
```

#### FetchContent

```cmake
include(FetchContent)

FetchContent_Declare(
    onpair
    GIT_REPOSITORY https://github.com/gargiulofrancesco/onpair_cpp.git
    GIT_TAG        <commit-hash>     # pin to a specific commit; main is unstable
)
FetchContent_MakeAvailable(onpair)

target_link_libraries(my_target PRIVATE OnPair::onpair)
```

When OnPair is consumed this way, tests, examples, install rules, and performance flags (`ONPAIR_ENABLE_LTO`, `ONPAIR_NATIVE_ARCH`) are all opt-in — the parent project keeps full control of its own build policy.

#### add_subdirectory

Drop OnPair into a sub-tree (vendored, submodule, or otherwise) and add it from the parent CMakeLists:

```cmake
add_subdirectory(third_party/onpair_cpp)
target_link_libraries(my_target PRIVATE OnPair::onpair)
```

Tests, examples, install rules, and performance flags default OFF in this mode.
