// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Selection-time candidate bookkeeping.

use vortex_array::ArrayRef;

use crate::estimate::EstimateScore;
use crate::scheme::Scheme;
use crate::scheme::SchemeId;

/// The mechanical facts the compressor knows about one (scheme, estimate) option during
/// scheme selection.
///
/// Selection currently reads only `scheme` and `score`; the remaining facts are carried so
/// that a pluggable cost model can price candidates without any new [`Scheme`] API
/// (see #7697).
pub(crate) struct Candidate {
    /// The scheme that produced this estimate.
    pub(crate) scheme: &'static dyn Scheme,

    /// The ranked estimate: an estimated or sample-measured compression ratio, or the
    /// zero-byte special case.
    pub(crate) score: EstimateScore,

    /// Uncompressed size in bytes of the array under selection.
    #[expect(dead_code, reason = "consumed by the cost model in a follow-up")]
    pub(crate) input_nbytes: u64,

    /// Number of values in the array under selection.
    #[expect(dead_code, reason = "consumed by the cost model in a follow-up")]
    pub(crate) n_values: u64,

    /// The compressed sample array, when this estimate came from sampling. Its encoding tree
    /// is the best available prediction of the full-array encoding tree. Dropped at the end
    /// of selection.
    #[expect(dead_code, reason = "consumed by the cost model in a follow-up")]
    pub(crate) sampled: Option<ArrayRef>,

    /// The cascade ancestry `(scheme_id, child_index)` of the selection site.
    #[expect(dead_code, reason = "consumed by the cost model in a follow-up")]
    pub(crate) cascade: Vec<(SchemeId, usize)>,
}
