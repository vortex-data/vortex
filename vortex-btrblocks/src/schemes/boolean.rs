// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression of non-nullable booleans, including validity arrays.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Bool;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::RunEnd;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_sparse::Sparse;
use vortex_sparse::SparseExt;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

/// Run-end encoding for boolean arrays with long runs.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BoolRunEndScheme;

impl Scheme for BoolRunEndScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.bool.runend"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype() == &DType::Bool(Nullability::NonNullable)
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![RunEnd.id()]
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
        let bits = data
            .array()
            .as_opt::<Bool>()
            .vortex_expect("boolean input")
            .to_bit_buffer();
        // Each true slice contributes at most two u64 ends and a final false run. Stop once
        // those ends alone would cost more than the bitmap.
        let runs = bits.set_slices().take(bits.len().div_ceil(128)).count() * 2 + 1;
        CompressionEstimate::Verdict(EstimateVerdict::Ratio(
            data.array().nbytes() as f64 / (runs as f64 * 8.125),
        ))
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let bits = data
            .array()
            .as_opt::<Bool>()
            .vortex_expect("boolean input")
            .to_bit_buffer();
        let mut ends = BufferMut::<u64>::empty();
        let mut values = Vec::new();
        let mut cursor = 0;
        for (start, end) in bits.set_slices() {
            if start > cursor {
                ends.push(start as u64);
                values.push(false);
            }
            ends.push(end as u64);
            values.push(true);
            cursor = end;
        }
        if cursor < bits.len() {
            ends.push(bits.len() as u64);
            values.push(false);
        }
        let ends = PrimitiveArray::new(ends.freeze(), Validity::NonNullable).narrow(exec_ctx)?;
        let ends =
            compressor.compress_child(&ends.into_array(), &compress_ctx, self.id(), 0, exec_ctx)?;
        Ok(RunEnd::try_new(
            ends,
            BoolArray::from_iter(values).into_array(),
            exec_ctx,
        )?
        .into_array())
    }
}

/// Sparse encoding for boolean arrays dominated by either true or false.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BoolSparseScheme;

impl Scheme for BoolSparseScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.bool.sparse"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype() == &DType::Bool(Nullability::NonNullable)
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Sparse.id()]
    }

    /// Children: indices=0.
    fn num_children(&self) -> usize {
        1
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let stats = data.bool_stats(exec_ctx);
        let exceptions = stats.true_count().min(stats.value_count() - stats.true_count());
        CompressionEstimate::Verdict(EstimateVerdict::Ratio(
            data.array().nbytes() as f64 / (exceptions as f64 * 4.0 + 2.0),
        ))
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let stats = data.bool_stats(exec_ctx);
        let fill = stats.true_count() > stats.value_count() / 2;
        let encoded = Sparse::encode(data.array(), Some(Scalar::from(fill)), exec_ctx)?;
        let Some(sparse) = encoded.as_opt::<Sparse>() else {
            return Ok(encoded);
        };
        let indices = sparse
            .patches()
            .indices()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?;
        let indices =
            compressor.compress_child(&indices.into_array(), &compress_ctx, self.id(), 0, exec_ctx)?;
        Sparse::try_new(
            indices,
            ConstantArray::new(!fill, sparse.patches().values().len()).into_array(),
            sparse.len(),
            sparse.fill_scalar().clone(),
            exec_ctx,
        )
        .map(IntoArray::into_array)
    }
}
