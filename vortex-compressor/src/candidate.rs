// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Selection-time candidate bookkeeping.

use vortex_array::ArrayRef;

use crate::scheme::EstimateScore;
use crate::scheme::Scheme;
use crate::scheme::SchemeId;

/// The mechanical facts the compressor knows about one (scheme, estimate) option during
/// scheme selection.
///
/// A `Candidate` is the argument to [`CostModel::cost`]: it carries everything a model can
/// price without any new [`Scheme`] API. Candidates are built by the compressor during
/// selection and dropped when selection ends.
///
/// [`CostModel::cost`]: crate::cost::CostModel::cost
#[derive(Debug)]
pub struct Candidate {
    /// The scheme that produced this estimate.
    pub scheme: &'static dyn Scheme,

    /// The ranked estimate: an estimated or sample-measured compression ratio, or the
    /// zero-byte special case.
    pub score: EstimateScore,

    /// Uncompressed size in bytes of the array under selection.
    pub input_nbytes: u64,

    /// Number of values in the array under selection.
    pub n_values: u64,

    /// The compressed sample array, when this estimate came from sampling. Its encoding tree
    /// is the best available prediction of the full-array encoding tree.
    pub sampled: Option<ArrayRef>,

    /// The cascade ancestry `(scheme_id, child_index)` of the selection site.
    pub cascade: Vec<(SchemeId, usize)>,
}
