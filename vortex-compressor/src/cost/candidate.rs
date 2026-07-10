// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Selection-time candidate view.

use vortex_array::ArrayRef;

use crate::scheme::CandidateEstimate;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::scheme::SchemeId;
use crate::stats::ArrayAndStats;

/// A borrowed view of one scheme estimate during selection.
///
/// The compressor constructs a `Candidate` immediately before calling [`CostModel::cost`].
/// It is valid only for that call: models can inspect it but cannot construct or retain it.
/// Accessors expose the stable, cheap facts available during selection without coupling the
/// public cost-model API to the selector's internal winner bookkeeping.
///
/// [`CostModel::cost`]: crate::cost::CostModel::cost
pub struct Candidate<'a> {
    /// The scheme that produced the estimate.
    scheme: &'static dyn Scheme,
    /// Model-independent evidence produced by the scheme.
    estimate: CandidateEstimate,
    /// The canonical input and its lazily generated statistics.
    data: &'a ArrayAndStats,
    /// The compressed sample for sampling-based estimates.
    sampled: Option<&'a ArrayRef>,
    /// The cascade ancestry of the selection site.
    cascade: &'a [(SchemeId, usize)],
}

impl<'a> Candidate<'a> {
    /// Creates a candidate view for one cost-model call.
    pub(crate) fn new(
        scheme: &'static dyn Scheme,
        estimate: CandidateEstimate,
        data: &'a ArrayAndStats,
        sampled: Option<&'a ArrayRef>,
        cascade: &'a [(SchemeId, usize)],
    ) -> Self {
        Self {
            scheme,
            estimate,
            data,
            sampled,
            cascade,
        }
    }

    /// Returns the ID of the scheme that produced the estimate.
    pub fn scheme_id(&self) -> SchemeId {
        self.scheme.id()
    }

    /// Returns the model-independent evidence produced by the scheme.
    pub fn estimate(&self) -> &CandidateEstimate {
        &self.estimate
    }

    /// Returns the canonical input array being selected over.
    pub fn array(&self) -> &ArrayRef {
        self.data.array()
    }

    /// Returns the uncompressed size of the input array in bytes.
    pub fn input_nbytes(&self) -> u64 {
        self.data.array().nbytes()
    }

    /// Returns the number of values in the input array.
    pub fn n_values(&self) -> u64 {
        self.data.array().len() as u64
    }

    /// Returns the compressed sample when this estimate came from sampling.
    ///
    /// The sample is borrowed only for the duration of the model call. Its encoding tree is
    /// the best available prediction of the full-array encoding tree.
    pub fn sampled(&self) -> Option<&ArrayRef> {
        self.sampled
    }

    /// Returns the cascade ancestry as `(scheme_id, child_index)` pairs.
    pub fn cascade(&self) -> &[(SchemeId, usize)] {
        self.cascade
    }
}
