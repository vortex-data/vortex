#pragma once
#include <onpair/core/types.h>
#include <cstdint>
#include <optional>
#include <variant>

namespace onpair::encoding {

// ─── Threshold Specifications ─────────────────────────────────────────────────

// Merge a token pair as soon as its frequency reaches `value`.
// Range: [2, 255].  The frequency counter is stored as uint8_t internally,
// so values above 255 cannot be represented and the type enforces this.
// In practice, compression ratio improves significantly at very low values
// but flattens rapidly beyond ~10.

struct FixedThreshold {
    uint8_t value;
};

// Adaptively tune the merge threshold so the dictionary fills to capacity
// within `sample_fraction` of the total input bytes. Values in (0.0, 1.0].
// 1.0 = train on the entire input; 0.5 = stop after processing half the bytes.
struct DynamicThreshold {
    double sample_fraction = 0.15;
};

using ThresholdSpec = std::variant<FixedThreshold, DynamicThreshold>;

// ─── Training Configuration ───────────────────────────────────────────────────

struct TrainingConfig {
    // Max dictionary size = 2^bits entries.  Legal range: [9, 16].
    BitWidth      bits            = 16;

    // Merge-frequency threshold.
    ThresholdSpec threshold       = DynamicThreshold{0.15};

    // RNG seed for the training shuffle.  nullopt → non-deterministic.
    // Set for reproducible compression (same dictionary across runs).
    std::optional<uint64_t> seed;
};

} // namespace onpair::encoding
