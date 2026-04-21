// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression context for recursive compression.

use std::fmt;

use vortex_error::VortexExpect;

use crate::compressor::ROOT_SCHEME_ID;
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
    merged_stats_options: GenerateStatsOptions,

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
            merged_stats_options: GenerateStatsOptions::default(),
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

    /// Returns the merged stats generation options for this compression site.
    pub fn merged_stats_options(&self) -> GenerateStatsOptions {
        self.merged_stats_options
    }

    /// Returns the cascade chain of `(scheme_id, child_index)` pairs.
    pub fn cascade_history(&self) -> &[(SchemeId, usize)] {
        &self.cascade_history
    }

    /// Returns a display wrapper for the current cascade ancestry.
    pub(crate) fn cascade_path(&self) -> impl fmt::Display + '_ {
        CascadePath(&self.cascade_history)
    }

    /// Returns the current cascade ancestry depth.
    pub(crate) fn cascade_depth(&self) -> usize {
        self.cascade_history.len()
    }

    /// Whether cascading is exhausted (no further cascade levels allowed).
    ///
    /// This should only be used in the implementation of a [`Scheme`](crate::scheme::Scheme) if the
    /// scheme knows that it's child _must_ be compressed for it to make any sense being chosen.
    pub fn finished_cascading(&self) -> bool {
        self.allowed_cascading == 0
    }

    /// Returns a context that disallows further cascading.
    pub fn as_leaf(mut self) -> Self {
        self.allowed_cascading = 0;
        self
    }

    /// Returns a context with the given stats options.
    pub(super) fn with_merged_stats_options(mut self, opts: GenerateStatsOptions) -> Self {
        self.merged_stats_options = opts;
        self
    }

    /// Returns a context marked as sample compression.
    pub(super) fn with_sampling(mut self) -> Self {
        self.is_sample = true;
        self
    }

    /// Descends one level in the cascade, recording the current scheme and which child is
    /// being compressed.
    ///
    /// The `child_index` identifies which child of the scheme is being compressed (e.g. for
    /// Dict: values=0, codes=1).
    pub(super) fn descend_with_scheme(mut self, id: SchemeId, child_index: usize) -> Self {
        self.allowed_cascading = self
            .allowed_cascading
            .checked_sub(1)
            .vortex_expect("cannot descend: cascade depth exhausted");
        self.cascade_history.push((id, child_index));
        self
    }
}

/// Display wrapper for a cascade ancestry path.
struct CascadePath<'a>(&'a [(SchemeId, usize)]);

impl fmt::Display for CascadePath<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return f.write_str("root");
        }

        for (index, (scheme_id, child_index)) in self.0.iter().enumerate() {
            if index > 0 {
                f.write_str(" > ")?;
            }

            if *scheme_id == ROOT_SCHEME_ID {
                write!(f, "root[{child_index}]")?;
            } else {
                write!(f, "{scheme_id}[{child_index}]")?;
            }
        }

        Ok(())
    }
}
