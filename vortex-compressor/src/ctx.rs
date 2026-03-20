// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression context for recursive compression.

use vortex_error::VortexExpect;

use crate::scheme::SchemeId;
use crate::stats::GenerateStatsOptions;

// TODO(connor): Why is this 3??? This doesn't seem smart or adaptive.
/// Maximum cascade depth for compression.
pub const MAX_CASCADE: usize = 3;

/// Context passed through recursive compression calls.
///
/// Tracks the cascade history (which schemes and child indices have been applied in the current
/// chain) so the compressor can enforce exclusion rules and prevent cycles.
#[derive(Debug, Clone)]
pub struct CompressorContext {
    /// Whether we're compressing a sample (for ratio estimation).
    is_sample: bool,
    /// Remaining cascade depth allowed.
    allowed_cascading: usize,
    /// Merged stats options from all eligible schemes at this compression site.
    stats_options: GenerateStatsOptions,
    /// The cascade chain: `(scheme_id, child_index)` pairs from root to current depth.
    /// Used for self-exclusion, push rules ([`descendant_exclusions`]), and pull rules
    /// ([`ancestor_exclusions`]).
    ///
    /// [`descendant_exclusions`]: crate::scheme::Scheme::descendant_exclusions
    /// [`ancestor_exclusions`]: crate::scheme::Scheme::ancestor_exclusions
    cascade_history: Vec<(SchemeId, usize)>,
}

impl CompressorContext {
    /// Creates a new `CompressorContext`.
    ///
    /// This should **only** be created by the compressor.
    pub(super) fn new() -> Self {
        Self {
            is_sample: false,
            allowed_cascading: MAX_CASCADE,
            stats_options: GenerateStatsOptions::default(),
            cascade_history: Vec::new(),
        }
    }
}

#[cfg(test)]
impl Default for CompressorContext {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressorContext {
    /// Whether this context is for sample compression (ratio estimation).
    pub fn is_sample(&self) -> bool {
        self.is_sample
    }

    /// Whether cascading is exhausted (no further cascade levels allowed).
    pub fn finished_cascading(&self) -> bool {
        self.allowed_cascading == 0
    }

    /// Returns the merged stats generation options for this compression site.
    pub fn stats_options(&self) -> GenerateStatsOptions {
        self.stats_options
    }

    /// Returns a context with the given stats options.
    pub fn with_stats_options(mut self, opts: GenerateStatsOptions) -> Self {
        self.stats_options = opts;
        self
    }

    /// Returns a context marked as sample compression.
    pub fn as_sample(mut self) -> Self {
        self.is_sample = true;
        self
    }

    /// Returns a context that disallows further cascading.
    pub fn as_leaf(mut self) -> Self {
        self.allowed_cascading = 0;
        self
    }

    /// Descends one level in the cascade, recording the current scheme and which child is
    /// being compressed.
    ///
    /// The `child_index` identifies which child of the scheme is being compressed (e.g. for
    /// Dict: values=0, codes=1).
    pub(crate) fn descend_with_scheme(mut self, id: SchemeId, child_index: usize) -> Self {
        self.allowed_cascading = self
            .allowed_cascading
            .checked_sub(1)
            .vortex_expect("cannot descend: cascade depth exhausted");
        self.cascade_history.push((id, child_index));
        self
    }

    /// Returns the cascade chain of `(scheme_id, child_index)` pairs.
    pub fn cascade_history(&self) -> &[(SchemeId, usize)] {
        &self.cascade_history
    }
}
