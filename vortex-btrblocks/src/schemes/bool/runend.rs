// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Run-end encoding for boolean arrays.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Bool;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::DeferredEstimate;
use vortex_compressor::estimate::EstimateVerdict;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_runend_bool::RunEndBool;
use vortex_runend_bool::RunEndBoolArrayExt;
use vortex_runend_bool::compress::encode_runend_bool;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

/// Minimum average run length before run-end encoding bool arrays is considered worthwhile.
const RUN_END_THRESHOLD: usize = 8;

/// Run-end encoding for boolean arrays.
///
/// Boolean runs strictly alternate, so the encoded form stores only the run ends (plus a `start`
/// flag and optional validity). This is profitable when runs are long.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BoolRunEndScheme;

impl BoolRunEndScheme {
    /// Count the number of boolean runs in the canonical bool array.
    fn run_count(data: &ArrayAndStats) -> usize {
        let bool_array = data
            .array()
            .as_opt::<Bool>()
            .vortex_expect("BoolRunEndScheme matches only canonical bool arrays");
        let bits = bool_array.to_bit_buffer();
        if bits.is_empty() {
            return 0;
        }
        // Each transition between true and false starts a new run; the number of runs is the number
        // of `true` slices plus the number of `false` gaps, which equals transitions + 1.
        let mut runs = 1usize;
        let mut prev = bits.value(0);
        for i in 1..bits.len() {
            let cur = bits.value(i);
            if cur != prev {
                runs += 1;
                prev = cur;
            }
        }
        runs
    }
}

impl Scheme for BoolRunEndScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.bool.runend"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches!(canonical, Canonical::Bool(_))
    }

    /// Children: ends=0.
    fn num_children(&self) -> usize {
        1
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

        let runs = Self::run_count(data).max(1);
        let average_run_length = length / runs;
        if average_run_length < RUN_END_THRESHOLD {
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
        let encoded = encode_runend_bool(bool_array, exec_ctx)?;

        let ends = encoded.ends().clone();
        let start = encoded.start();
        let offset = encoded.offset();
        let validity = encoded.bool_validity();
        let length = encoded.as_ref().len();

        let ends_primitive = ends.execute::<PrimitiveArray>(exec_ctx)?;
        let compressed_ends = compressor.compress_child(
            &ends_primitive.into_array(),
            &compress_ctx,
            self.id(),
            0,
            exec_ctx,
        )?;

        // SAFETY: compression of the ends preserves the strictly-increasing invariant.
        Ok(unsafe {
            RunEndBool::new_unchecked(compressed_ends, start, offset, length, validity).into_array()
        })
    }
}
