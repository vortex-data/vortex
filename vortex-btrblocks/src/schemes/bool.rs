// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bool compression schemes.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
pub use vortex_compressor::builtins::BoolConstantScheme;
use vortex_compressor::builtins::IntDictScheme;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::EstimateVerdict;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::DescendantExclusion;
pub use vortex_compressor::stats::BoolStats;
use vortex_runend::RunEnd;
use vortex_runend::compress::runend_encode_bool;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::integer::IntRLEScheme;
use crate::schemes::integer::RunEndScheme;
use crate::schemes::integer::SparseScheme;

/// Run-end encoding for bool arrays.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BoolRunEndScheme;

// Cheap fast-skip: below this average run length, REE's ends array is unlikely to beat
// bit-packed bool values. The exact byte estimate below remains the final gate.
const BOOL_RUN_END_THRESHOLD: usize = 8;

impl Scheme for BoolRunEndScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.bool.runend"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches!(canonical, Canonical::Bool(_))
    }

    /// Children: values=0, ends=1.
    fn num_children(&self) -> usize {
        2
    }

    /// RunEnd ends (child 1) are monotonically increasing positions with all unique values.
    /// Dict, RunEnd, RLE, and Sparse are all pointless on such data.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![
            DescendantExclusion {
                excluded: IntDictScheme.id(),
                children: ChildSelection::One(1),
            },
            DescendantExclusion {
                excluded: RunEndScheme.id(),
                children: ChildSelection::One(1),
            },
            DescendantExclusion {
                excluded: IntRLEScheme.id(),
                children: ChildSelection::One(1),
            },
            DescendantExclusion {
                excluded: SparseScheme.id(),
                children: ChildSelection::One(1),
            },
        ]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let stats = data.bool_stats(exec_ctx);
        let run_count = stats.run_count() as usize;

        if run_count == 0 || data.array_len() < run_count.saturating_mul(BOOL_RUN_END_THRESHOLD) {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // The encoder only materializes run-value validity when there are actual null runs.
        // Nullable bool arrays with no nulls use an all-valid values child.
        let has_null_runs = stats.null_count() > 0;
        let before_nbytes = data.array().nbytes();
        let after_nbytes =
            estimated_runend_bool_nbytes(data.array_len(), run_count, has_null_runs) as u64;

        if after_nbytes >= before_nbytes {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        CompressionEstimate::Verdict(EstimateVerdict::Ratio(
            before_nbytes as f64 / after_nbytes as f64,
        ))
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> vortex_error::VortexResult<ArrayRef> {
        let (ends, values) = runend_encode_bool(data.array_as_bool(), exec_ctx);

        let compressed_values =
            compressor.compress_child(&values, &compress_ctx, self.id(), 0, exec_ctx)?;
        let compressed_ends =
            compressor.compress_child(&ends.into_array(), &compress_ctx, self.id(), 1, exec_ctx)?;

        // SAFETY: compression doesn't affect invariants.
        Ok(unsafe {
            RunEnd::new_unchecked(compressed_ends, compressed_values, 0, data.array_len())
                .into_array()
        })
    }
}

fn estimated_runend_bool_nbytes(len: usize, run_count: usize, nullable: bool) -> usize {
    let ends_nbytes = run_count * run_end_width(len);
    let values_nbytes = run_count.div_ceil(8);
    let validity_nbytes = if nullable { run_count.div_ceil(8) } else { 0 };

    ends_nbytes + values_nbytes + validity_nbytes
}

fn run_end_width(len: usize) -> usize {
    if u8::try_from(len).is_ok() {
        size_of::<u8>()
    } else if u16::try_from(len).is_ok() {
        size_of::<u16>()
    } else if u32::try_from(len).is_ok() {
        size_of::<u32>()
    } else {
        size_of::<u64>()
    }
}
