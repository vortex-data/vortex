// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Port of `include/onpair/encoding/training/config.h` plus the
// FFI-shaped `OnPairTrainingConfig` from `vortex-onpair-sys`.

use crate::types::BitWidth;

/// Merge a token pair as soon as its frequency reaches `value`.
///
/// Range: `[2, 255]`. The frequency counter is `u8` so larger values can
/// never trigger a merge.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FixedThreshold {
    pub value: u8,
}

/// Adaptively tune the merge threshold so the dictionary fills to capacity
/// within `sample_fraction` of the total input bytes. Values in `(0.0, 1.0]`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DynamicThreshold {
    pub sample_fraction: f64,
}

impl Default for DynamicThreshold {
    fn default() -> Self {
        Self { sample_fraction: 0.15 }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum ThresholdSpec {
    Fixed(FixedThreshold),
    Dynamic(DynamicThreshold),
}

impl Default for ThresholdSpec {
    fn default() -> Self {
        Self::Dynamic(DynamicThreshold::default())
    }
}

/// Internal, full-fidelity training config matching the C++
/// `encoding::TrainingConfig`.
#[derive(Clone, Debug)]
pub struct TrainingConfig {
    /// `2^bits` is the max dictionary size. Legal range: `9..=16`.
    pub bits: BitWidth,
    pub threshold: ThresholdSpec,
    /// `None` → non-deterministic seed.
    pub seed: Option<u64>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            bits: 16,
            threshold: ThresholdSpec::default(),
            seed: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FFI-shaped config.
//
// Matches `vortex-onpair-sys::OnPairTrainingConfig` field-for-field so this
// crate can be a drop-in replacement. `seed == 0` is interpreted as
// "non-deterministic", same as the C shim's behaviour.
// ─────────────────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct OnPairTrainingConfig {
    pub bits: u32,
    pub threshold: f64,
    pub seed: u64,
}

/// `dict-12`: 12-bit codes (4 096 entries), dynamic threshold 0.5.
pub const DEFAULT_DICT12_CONFIG: OnPairTrainingConfig = OnPairTrainingConfig {
    bits: 12,
    threshold: 0.5,
    seed: 0,
};

impl From<OnPairTrainingConfig> for TrainingConfig {
    fn from(c: OnPairTrainingConfig) -> Self {
        Self {
            bits: c.bits as BitWidth,
            threshold: ThresholdSpec::Dynamic(DynamicThreshold {
                sample_fraction: c.threshold,
            }),
            seed: (c.seed != 0).then_some(c.seed),
        }
    }
}

/// Errors mirroring `vortex-onpair-sys::Error`.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Error {
    InvalidArg,
    BadFormat,
    OutOfRange,
    Oom,
    Internal,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            Error::InvalidArg => "OnPair: invalid argument",
            Error::BadFormat => "OnPair: bad serialized format",
            Error::OutOfRange => "OnPair: row index out of range",
            Error::Oom => "OnPair: out of memory or buffer too small",
            Error::Internal => "OnPair: internal error",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for Error {}
