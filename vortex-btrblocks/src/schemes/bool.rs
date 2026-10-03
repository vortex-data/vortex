// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Run-end encoding for booleans, such as the validity of sparse columns.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_runend::RunEnd;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

/// Run-end encoding of a non-nullable boolean array as alternating runs.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BoolRunEndScheme;

/// The ends of the runs of equal bits and the value of the first run.
fn bool_runs(bits: &BitBuffer) -> (Vec<u64>, bool) {
    let len = bits.len();
    let mut ends = Vec::new();
    let mut cursor = 0;
    for (start, end) in bits.set_slices() {
        if start > cursor {
            ends.push(start as u64);
        }
        ends.push(end as u64);
        cursor = end;
    }
    if cursor < len {
        ends.push(len as u64);
    }
    let first = len > 0 && bits.value(0);
    (ends, first)
}

impl Scheme for BoolRunEndScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.bool.runend"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches!(canonical, Canonical::Bool(b) if !b.dtype().is_nullable())
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![RunEnd.id()]
    }

    /// Children: values=0, ends=1.
    fn num_children(&self) -> usize {
        2
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let Ok(bools) = data.array().clone().execute::<BoolArray>(exec_ctx) else {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        };
        let bits = bools.to_bit_buffer();
        let runs = bits.set_slices().count() * 2 + 1;
        // Before compressing the ends, each run costs a 32-bit end plus one value bit.
        #[allow(clippy::cast_precision_loss)]
        let ratio = (bits.len() as f64 / 8.0) / (runs as f64 * 4.125);
        if ratio > 1.0 {
            CompressionEstimate::Verdict(EstimateVerdict::Ratio(ratio))
        } else {
            CompressionEstimate::Verdict(EstimateVerdict::Skip)
        }
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let bits = data.array().clone().execute::<BoolArray>(exec_ctx)?.to_bit_buffer();
        let (ends, first) = bool_runs(&bits);
        let mut values = BitBufferMut::with_capacity(ends.len());
        for i in 0..ends.len() {
            values.append(first ^ (i % 2 == 1));
        }
        let values = BoolArray::new(values.freeze(), Validity::NonNullable).into_array();
        let ends = PrimitiveArray::new(Buffer::from(ends), Validity::NonNullable).into_array();
        let ends = compressor.compress_child(&ends, &compress_ctx, self.id(), 1, exec_ctx)?;

        // SAFETY: the ends are strictly increasing and the last one is the array length.
        Ok(unsafe { RunEnd::new_unchecked(ends, values, 0, data.array_len()).into_array() })
    }
}
