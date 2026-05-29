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

/// Derive the per-block SORF seed from the global TurboQuant seed and the block index.
///
/// The derivation offsets `global_seed` by `block_index * SPLITMIX64_INCREMENT` (matching the
/// additive part of one splitmix64 step) and then applies the splitmix64 mixing tail (the two
/// `MUL1` / `MUL2` rounds plus the final xor-shift). `block_index = 0` is therefore the mixing
/// tail applied directly to `global_seed`, not `global_seed` itself.
///
/// This function is part of the wire contract and MUST NOT change once shipped: the per-block
/// sign mask stream depends on this output exactly.
pub(crate) fn derive_block_seed(global_seed: u64, block_index: usize) -> u64 {
    // `usize::MAX <= u64::MAX` on every target this crate supports, so the cast is lossless.
    let block_index = block_index as u64;
    let mut state = global_seed.wrapping_add(block_index.wrapping_mul(SPLITMIX64_INCREMENT));
    state = (state ^ (state >> 30)).wrapping_mul(SPLITMIX64_MUL1);
    state = (state ^ (state >> 27)).wrapping_mul(SPLITMIX64_MUL2);
    state ^ (state >> 31)
}

#[cfg(test)]
mod tests {
    use super::SplitMix64;
    use super::derive_block_seed;

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

    /// Golden values for `derive_block_seed(42, 0..4)` computed by hand from the splitmix64
    /// reference (additive offset of `block_index * INCREMENT`, followed by the two MUL rounds
    /// and the final xor-shift). These pin the wire contract.
    ///
    /// Indices 1, 2, and 3 align with `SPLITMIX64_SEED42_GOLDEN[0..3]` because `SplitMix64`'s
    /// `next_u64` increments before mixing: its `k`-th output is `mix(seed + (k + 1) * INCREMENT)`,
    /// while `derive_block_seed(seed, k)` is `mix(seed + k * INCREMENT)`. Index 0 is the mixing
    /// tail applied directly to `42`, which has no counterpart in the existing stream golden.
    const DERIVED_SEED_42_GOLDEN: [u64; 4] = [
        0xA759_EA27_D472_7622,
        0xBDD7_3226_2FEB_6E95,
        0x28EF_E333_B266_F103,
        0x4752_6757_130F_9F52,
    ];

    #[test]
    fn derive_block_seed_matches_splitmix64_stream_at_zero_indices() {
        let actual: Vec<u64> = (0..DERIVED_SEED_42_GOLDEN.len())
            .map(|i| derive_block_seed(42, i))
            .collect();
        assert_eq!(actual, DERIVED_SEED_42_GOLDEN);
    }

    #[test]
    fn derive_block_seed_distinct_for_consecutive_indices() {
        let mut seeds: Vec<u64> = (0..16).map(|i| derive_block_seed(0xDEAD_BEEF, i)).collect();
        seeds.sort_unstable();
        seeds.dedup();
        assert_eq!(seeds.len(), 16, "derive_block_seed produced duplicates");
    }
}
