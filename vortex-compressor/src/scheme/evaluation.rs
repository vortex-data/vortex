// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scheme evaluation and model-independent candidate evidence.

use std::fmt;

use vortex_array::ExecutionCtx;
use vortex_error::VortexResult;

use crate::CascadingCompressor;
use crate::scheme::CompressorContext;
use crate::stats::ArrayAndStats;

/// Closure type for [`DeferredEvaluation::Callback`].
///
/// The compressor invokes this when a scheme needs fallible or expensive work to produce a
/// terminal [`ResolvedEvaluation`]. Candidate comparison is deliberately absent from this API:
/// the callback describes the candidate, and the configured cost model decides whether it wins.
#[rustfmt::skip]
pub type DeferredEvaluationFn = dyn FnOnce(
        &CascadingCompressor,
        &ArrayAndStats,
        CompressorContext,
        &mut ExecutionCtx,
    ) -> VortexResult<ResolvedEvaluation>
    + Send
    + Sync;

/// A scheme's initial evaluation result.
///
/// A scheme either declines the input, forces itself for semantic reasons, produces a candidate
/// estimate immediately, or asks the compressor to resolve deferred work. Candidate estimates are
/// model-independent facts; the compressor wraps them in a [`Candidate`](crate::cost::Candidate)
/// and passes them to the configured cost model.
#[derive(Debug)]
pub enum SchemeEvaluation {
    /// Do not consider this scheme for the input.
    Skip,

    /// Select this scheme immediately without cost-model comparison.
    AlwaysUse,

    /// Hand this estimate to the configured cost model as a candidate.
    Candidate(CandidateEstimate),

    /// The compressor must perform deferred work to resolve a terminal result.
    Deferred(DeferredEvaluation),
}

/// A terminal result produced by a deferred scheme evaluation.
#[derive(Debug)]
pub enum ResolvedEvaluation {
    /// Do not consider this scheme for the input.
    Skip,

    /// Select this scheme immediately without cost-model comparison.
    ///
    /// Some examples include decimal byte parts and temporal decomposition.
    ///
    /// The compressor will select this scheme immediately without evaluating further candidates.
    /// Schemes that return `AlwaysUse` must be mutually exclusive per canonical type (enforced by
    /// [`Scheme::matches`]), otherwise the winner depends silently on registration order.
    ///
    /// [`Scheme::matches`]: crate::scheme::Scheme::matches
    AlwaysUse,

    /// Hand this estimate to the configured cost model as a candidate.
    Candidate(CandidateEstimate),
}

/// Deferred work that can produce a terminal [`ResolvedEvaluation`].
pub enum DeferredEvaluation {
    /// Compress a small sample and expose the resulting candidate to the cost model.
    Sample,

    /// Run a scheme-defined fallible or expensive evaluation.
    ///
    /// The callback returns a [`ResolvedEvaluation`] directly, so it cannot request more sampling
    /// or another deferred callback.
    Callback(Box<DeferredEvaluationFn>),
}

/// Model-independent evidence produced by a scheme for one candidate.
///
/// The estimate is intentionally opaque and evolution-safe. Compression ratio is currently the
/// common signal produced by schemes and consumed by [`SizeCost`](crate::cost::SizeCost); it is not
/// an ordering used by the compressor itself.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateEstimate {
    /// Estimated compression ratio, absent when a sampled candidate produced zero bytes.
    estimated_compression_ratio: Option<f64>,
}

impl CandidateEstimate {
    /// Creates candidate evidence from an estimated compression ratio.
    pub fn from_compression_ratio(ratio: f64) -> Self {
        Self {
            estimated_compression_ratio: Some(ratio),
        }
    }

    /// Returns the estimated compression ratio, or `None` when a sample produced zero bytes.
    pub fn estimated_compression_ratio(&self) -> Option<f64> {
        self.estimated_compression_ratio
    }

    /// Creates candidate evidence for a sampled output containing zero bytes.
    pub(crate) fn zero_bytes() -> Self {
        Self {
            estimated_compression_ratio: None,
        }
    }
}

impl fmt::Debug for DeferredEvaluation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeferredEvaluation::Sample => write!(f, "Sample"),
            DeferredEvaluation::Callback(_) => write!(f, "Callback(..)"),
        }
    }
}
