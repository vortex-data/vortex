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
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::RunEnd;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::runend::compress::bool_run_count;
use vortex_array::arrays::runend::compress::runend_encode_bool;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_compressor::builtins::IntDictScheme;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_sparse::Sparse;
use vortex_sparse::SparseExt;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::Child;
use crate::CompressorContext;
use crate::Encoded;
use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::integer::IntRLEScheme;
use crate::schemes::integer::RunEndScheme;
use crate::schemes::integer::SparseScheme as IntSparseScheme;

/// Estimated bytes per run for `BoolRunEndScheme`: narrowed 1-based end (`1..=len`) plus 1 bit.
fn runend_bytes_per_run(len: usize) -> f64 {
    PType::min_unsigned_ptype_for_value(len as u64).byte_width() as f64 + 0.125
}

/// Byte width of narrowed 0-based patch indices (`0..len`) for `BoolSparseScheme`.
fn sparse_index_width(len: usize) -> usize {
    PType::min_unsigned_ptype_for_value(len.saturating_sub(1) as u64).byte_width()
}

/// Descendant exclusions for strictly increasing, unique position children at child index 0.
fn monotonic_index_exclusions() -> Vec<DescendantExclusion> {
    vec![
        DescendantExclusion {
            excluded: IntDictScheme.id(),
            children: ChildSelection::One(0),
        },
        DescendantExclusion {
            excluded: RunEndScheme.id(),
            children: ChildSelection::One(0),
        },
        DescendantExclusion {
            excluded: IntRLEScheme.id(),
            children: ChildSelection::One(0),
        },
        DescendantExclusion {
            excluded: IntSparseScheme.id(),
            children: ChildSelection::One(0),
        },
    ]
}

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

    /// RunEnd ends (child 0) are strictly increasing positions with all unique values.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        monotonic_index_exclusions()
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
        // `runend_encode_bool` narrows 1-based ends (`1..=len`) to the tightest unsigned type and
        // stores one value bit per run.
        let runs = bool_run_count(&bits).max(1);
        let bytes_per_run = runend_bytes_per_run(data.array_len());
        CompressionEstimate::Verdict(EstimateVerdict::Ratio(
            data.array().nbytes() as f64 / (runs as f64 * bytes_per_run),
        ))
    }

    /// Children: ends=0, and the run values, kept at 1: one bit per run.
    fn encode(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Encoded> {
        let array = data.array().as_opt::<Bool>().vortex_expect("boolean input");
        let (ends, values) = runend_encode_bool(array, exec_ctx)?;
        Ok(Encoded::new(
            vec![Child::new(self, 0, ends.into_array())],
            vec![Child::new(self, 1, values)],
            |children, exec_ctx| {
                let [ends] = children.compressed()?;
                let [values] = children.kept()?;
                Ok(RunEnd::try_new(ends, values, exec_ctx)?.into_array())
            },
        ))
    }

    #[expect(clippy::unimplemented, reason = "replaced by `encode`")]
    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unimplemented!("{} compresses through Scheme::encode", self.scheme_name())
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
        vec![Sparse.id(), Constant.id()]
    }

    /// Children: indices=0.
    fn num_children(&self) -> usize {
        1
    }

    /// Sparse indices (child 0) are strictly increasing positions with all unique values.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        monotonic_index_exclusions()
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let stats = data.bool_stats(exec_ctx);
        let exceptions = stats
            .true_count()
            .min(stats.value_count() - stats.true_count());
        // `encode` narrows 0-based indices (`0..len`) to the tightest unsigned type.
        let index_width = sparse_index_width(data.array_len());
        CompressionEstimate::Verdict(EstimateVerdict::Ratio(
            data.array().nbytes() as f64 / (exceptions as f64 * index_width as f64 + 2.0),
        ))
    }

    /// Children: indices=0. Every patch value is the exception, so the values are a constant.
    fn encode(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Encoded> {
        let stats = data.bool_stats(exec_ctx);
        let fill = stats.true_count() > stats.value_count() / 2;
        let encoded = Sparse::encode(data.array(), Some(Scalar::from(fill)), exec_ctx)?;
        let Some(sparse) = encoded.as_opt::<Sparse>() else {
            return Ok(Encoded::leaf(encoded));
        };
        let indices = sparse
            .patches()
            .indices()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?;
        let num_patches = sparse.patches().values().len();
        let len = sparse.len();
        let fill_scalar = sparse.fill_scalar().clone();
        Ok(Encoded::new(
            vec![Child::new(self, 0, indices.into_array())],
            Vec::new(),
            move |children, exec_ctx| {
                let [indices] = children.compressed()?;
                Sparse::try_new(
                    indices,
                    ConstantArray::new(!fill, num_patches).into_array(),
                    len,
                    fill_scalar,
                    exec_ctx,
                )
                .map(IntoArray::into_array)
            },
        ))
    }

    #[expect(clippy::unimplemented, reason = "replaced by `encode`")]
    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unimplemented!("{} compresses through Scheme::encode", self.scheme_name())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::RunEnd;
    use vortex_array::assert_arrays_eq;
    use vortex_compressor::builtins::IntDictScheme;
    use vortex_compressor::scheme::ChildSelection;
    use vortex_error::VortexResult;
    use vortex_sparse::Sparse;

    use super::BoolRunEndScheme;
    use super::BoolSparseScheme;
    use super::runend_bytes_per_run;
    use super::sparse_index_width;
    use crate::BtrBlocksCompressorBuilder;
    use crate::SESSION;
    use crate::Scheme;
    use crate::SchemeExt;
    use crate::schemes::integer::IntRLEScheme;
    use crate::schemes::integer::RunEndScheme;
    use crate::schemes::integer::SparseScheme as IntSparseScheme;

    #[test]
    fn test_bool_runend_compresses_medium_runs() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let input = BoolArray::from_iter((0..4_096).map(|i| (i / 24) % 2 == 0)).into_array();
        let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
            .unrestricted()
            .build();

        let compressed = compressor.compress(&input, &mut ctx)?;
        assert!(
            compressed.is::<RunEnd>(),
            "expected RunEnd, got {}",
            compressed.encoding_id()
        );
        assert!(compressed.nbytes() < input.nbytes());
        assert_arrays_eq!(compressed, input, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_bool_sparse_prices_u16_indices_at_65536_rows() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // 2,979 scattered exceptions in 65,536 rows: 8,192 B raw vs ~5,958 B with u16 indices.
        // Hardcoding u32 (4 B/patch) priced this at ~11,918 B (< 1.0x) and left it uncompressed.
        let input = BoolArray::from_iter((0..65_536).map(|i| i % 22 == 0)).into_array();
        let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
            .unrestricted()
            .build();

        let compressed = compressor.compress(&input, &mut ctx)?;
        assert!(
            compressed.is::<Sparse>(),
            "expected Sparse, got {}",
            compressed.encoding_id()
        );
        assert!(compressed.nbytes() < input.nbytes());
        assert_arrays_eq!(compressed, input, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::u8_max(255, 1.125, 1)]
    #[case::u8_to_u16(256, 2.125, 1)]
    #[case::u16_min_sparse(257, 2.125, 2)]
    #[case::u16_max(65_535, 2.125, 2)]
    #[case::u16_to_u32(65_536, 4.125, 2)]
    #[case::u32_min_sparse(65_537, 4.125, 4)]
    fn test_bool_index_width_boundaries(
        #[case] len: usize,
        #[case] expected_runend_bytes: f64,
        #[case] expected_sparse_width: usize,
    ) {
        assert!((runend_bytes_per_run(len) - expected_runend_bytes).abs() < 1e-12);
        assert_eq!(sparse_index_width(len), expected_sparse_width);
    }

    #[test]
    fn test_bool_schemes_exclude_pointless_index_schemes() {
        let expected = [
            IntDictScheme.id(),
            RunEndScheme.id(),
            IntRLEScheme.id(),
            IntSparseScheme.id(),
        ];

        for scheme in [&BoolRunEndScheme as &dyn Scheme, &BoolSparseScheme] {
            let exclusions = scheme.descendant_exclusions();
            for id in expected {
                assert!(
                    exclusions.iter().any(|rule| {
                        rule.excluded == id && matches!(rule.children, ChildSelection::One(0))
                    }),
                    "{} missing child-0 exclusion for {id}",
                    scheme.scheme_name()
                );
            }
        }
    }
}
