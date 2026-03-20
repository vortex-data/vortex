// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression statistics types.

/// Configures how stats are generated.
///
/// Each scheme declares its required options via [`Scheme::stats_options`]. The compressor
/// merges all eligible schemes' options before generating stats, so that a single stats pass
/// satisfies every scheme.
///
/// [`Scheme::stats_options`]: crate::scheme::Scheme::stats_options
#[derive(Debug, Default, Clone, Copy)]
pub struct GenerateStatsOptions {
    /// Whether distinct values should be counted during stats generation.
    pub count_distinct_values: bool,
}

impl GenerateStatsOptions {
    /// Merges two options by OR-ing each field. The result enables a stat if either input does.
    pub fn merge(self, other: Self) -> Self {
        Self {
            count_distinct_values: self.count_distinct_values || other.count_distinct_values,
        }
    }
}
