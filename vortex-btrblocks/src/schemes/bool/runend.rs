// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Run-end encoding for boolean columns via the generic [`RunEnd`] encoding.
//!
//! Boolean runs strictly alternate, so a run-end encoded bool array is a generic
//! `RunEnd{ends, values}` whose values child is the per-run boolean array. A future
//! cycling/repeat-pattern array encoding could stand in as that values child to recover
//! single-`start`-bit compactness through composition — see [`runend_encode_bool`].

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Bool;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_runend::RunEnd;
use vortex_runend::compress::runend_encode_bool;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::integer::RunEndScheme;

/// Minimum average run length before run-end encoding a boolean column is considered worthwhile.
const RUN_END_THRESHOLD: usize = 4;

/// Run-end encoding for boolean columns.
///
/// Emits a generic [`RunEnd`] array whose values child is the per-run boolean array; the run
/// `ends` (child 1) are cascaded for further compression.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BoolRunEndScheme;

impl BoolRunEndScheme {
    /// Count the number of runs in the canonical boolean array.
    ///
    /// Uses the word-at-a-time `set_slices` iterator rather than a per-bit scan.
    fn run_count(array: &ArrayRef) -> usize {
        let bool_array = array
            .as_opt::<Bool>()
            .vortex_expect("BoolRunEndScheme matches only canonical bool arrays");
        let bits = bool_array.to_bit_buffer();
        let len = bits.len();
        if len == 0 {
            return 0;
        }

        // Each `true` slice is one run; a `false` gap before it or a trailing `false` gap is
        // another. `set_slices` yields the maximal `true` ranges word-at-a-time.
        let mut runs = 0usize;
        let mut cursor = 0usize;
        for (start, end) in bits.set_slices() {
            if start > cursor {
                runs += 1; // leading/interior false gap
            }
            runs += 1; // the true run
            cursor = end;
        }
        if cursor < len {
            runs += 1; // trailing false gap
        }
        runs
    }
}

impl Scheme for BoolRunEndScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.bool.runend"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_boolean()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![RunEnd.id()]
    }

    /// Children: values=0, ends=1.
    fn num_children(&self) -> usize {
        2
    }

    /// RunEnd ends (child 1) are monotonically increasing positions with all unique values, so
    /// run-end encoding them again is pointless.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![DescendantExclusion {
            excluded: RunEndScheme.id(),
            children: ChildSelection::One(1),
        }]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let length = data.array_len();
        if length == 0 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        let runs = Self::run_count(data.array()).max(1);
        if length / runs < RUN_END_THRESHOLD {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let bool_array = data
            .array()
            .as_opt::<Bool>()
            .vortex_expect("BoolRunEndScheme matches only canonical bool arrays");
        let length = data.array_len();

        let (ends, values) = runend_encode_bool(bool_array, exec_ctx);

        let compressed_values =
            compressor.compress_child(&values, &compress_ctx, self.id(), 0, exec_ctx)?;

        let compressed_ends =
            compressor.compress_child(&ends.into_array(), &compress_ctx, self.id(), 1, exec_ctx)?;

        // SAFETY: compression preserves the strictly-increasing ends invariant.
        Ok(unsafe {
            RunEnd::new_unchecked(compressed_ends, compressed_values, 0, length).into_array()
        })
    }
}
