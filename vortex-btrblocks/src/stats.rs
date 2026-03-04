// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression statistics types.

use std::fmt::Debug;

use vortex_array::vtable::VTable;

/// Configures how stats are generated.
#[derive(Clone, Copy)]
pub struct GenerateStatsOptions {
    /// Should distinct values should be counted during stats generation.
    pub count_distinct_values: bool,
    // pub count_runs: bool,
    // should this be scheme-specific?
}

impl Default for GenerateStatsOptions {
    fn default() -> Self {
        Self {
            count_distinct_values: true,
            // count_runs: true,
        }
    }
}

/// The size of each sampled run.
pub(crate) const SAMPLE_SIZE: u32 = 64;
/// The number of sampled runs.
///
/// # Warning
///
/// The product of SAMPLE_SIZE and SAMPLE_COUNT should be (roughly) a multiple of 1024 so that
/// fastlanes bitpacking of sampled vectors does not introduce (large amounts of) padding.
pub(crate) const SAMPLE_COUNT: u32 = 16;

/// Stats for the compressor.
pub trait CompressorStats: Debug + Clone {
    /// The type of the underlying source array vtable.
    type ArrayVTable: VTable;

    /// Generates stats with default options.
    fn generate(input: &<Self::ArrayVTable as VTable>::Array) -> Self {
        Self::generate_opts(input, GenerateStatsOptions::default())
    }

    /// Generates stats with provided options.
    fn generate_opts(
        input: &<Self::ArrayVTable as VTable>::Array,
        opts: GenerateStatsOptions,
    ) -> Self;

    /// Returns the underlying source array that statistics were generated from.
    fn source(&self) -> &<Self::ArrayVTable as VTable>::Array;

    /// Sample the array with default options.
    fn sample(&self, sample_size: u32, sample_count: u32) -> Self {
        self.sample_opts(sample_size, sample_count, GenerateStatsOptions::default())
    }

    /// Sample the array with provided options.
    fn sample_opts(&self, sample_size: u32, sample_count: u32, opts: GenerateStatsOptions) -> Self;
}
