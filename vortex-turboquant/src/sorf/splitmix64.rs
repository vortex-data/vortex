// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Frozen local SplitMix64 stream used to define SORF sign diagonals.
//!
//! This is a direct translation of the `splitmix64.c` [reference implementation][impl].
//!
//! The state is a single `u64`, and `next_u64()` first adds [`SPLITMIX64_INCREMENT`] with wrapping
//! arithmetic, then applies the two reference mixing steps and final xor-shift.
//!
//! [impl]: https://prng.di.unimi.it/splitmix64.c

/// SplitMix64 additive constant from the reference implementation.
const SPLITMIX64_INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;

/// First SplitMix64 mixing multiplier from the reference implementation.
const SPLITMIX64_MUL1: u64 = 0xBF58_476D_1CE4_E5B9;

/// Second SplitMix64 mixing multiplier from the reference implementation.
const SPLITMIX64_MUL2: u64 = 0x94D0_49BB_1331_11EB;

/// Frozen local SplitMix64 stream used to define SORF sign diagonals. Bit-identical to the
/// reference implementation linked at the module top, which makes the sign stream part of the
/// encoding's wire contract.
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(SPLITMIX64_INCREMENT);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(SPLITMIX64_MUL1);
        z = (z ^ (z >> 27)).wrapping_mul(SPLITMIX64_MUL2);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::SplitMix64;

    const SPLITMIX64_SEED0_GOLDEN: [u64; 4] = [
        0xE220_A839_7B1D_CDAF,
        0x6E78_9E6A_A1B9_65F4,
        0x06C4_5D18_8009_454F,
        0xF88B_B8A8_724C_81EC,
    ];

    const SPLITMIX64_SEED42_GOLDEN: [u64; 4] = [
        0xBDD7_3226_2FEB_6E95,
        0x28EF_E333_B266_F103,
        0x4752_6757_130F_9F52,
        0x581C_E1FF_0E4A_E394,
    ];

    #[test]
    fn splitmix64_seed0_matches_golden_outputs() {
        let mut rng = SplitMix64::new(0);
        let actual: Vec<_> = (0..SPLITMIX64_SEED0_GOLDEN.len())
            .map(|_| rng.next_u64())
            .collect();
        assert_eq!(actual, SPLITMIX64_SEED0_GOLDEN);
    }

    #[test]
    fn splitmix64_seed42_matches_golden_outputs() {
        let mut rng = SplitMix64::new(42);
        let actual: Vec<_> = (0..SPLITMIX64_SEED42_GOLDEN.len())
            .map(|_| rng.next_u64())
            .collect();
        assert_eq!(actual, SPLITMIX64_SEED42_GOLDEN);
    }
}
