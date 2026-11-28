// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sparse integer encoding for single-value-dominated arrays.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
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
use vortex_sparse::SparseExt as _;

use super::IntRLEScheme;
use super::RunEndScheme;
use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::Child;
use crate::CompressorContext;
use crate::Encoded;
use crate::GenerateStatsOptions;
use crate::Scheme;
use crate::SchemeExt;

/// Sparse encoding for single-value-dominated arrays.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SparseScheme;

impl Scheme for SparseScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.sparse"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Sparse.id(), Constant.id()]
    }

    fn stats_options(&self) -> GenerateStatsOptions {
        GenerateStatsOptions {
            count_distinct_values: true,
        }
    }

    /// Children: values=0, indices=1.
    fn num_children(&self) -> usize {
        2
    }

    /// Sparse indices (child 1) are monotonically increasing positions with all unique values.
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
        let len = data.array_len();
        let stats = data.integer_stats(exec_ctx);
        let value_count = stats.value_count();

        // All-null arrays should be compressed as constant instead anyways.
        if value_count == 0 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        let elem_bytes = data.array_as_primitive().ptype().byte_width();
        let idx_width =
            PType::min_unsigned_ptype_for_value(len.saturating_sub(1) as u64).byte_width();
        let has_nulls = stats.null_count() > 0;
        let uncompressed_bytes = len * elem_bytes + if has_nulls { len.div_ceil(8) } else { 0 };

        // If more than 90% of values are null, `Sparse::encode` uses a null fill value and stores
        // only the non-null values (no patch validity bitmap) plus narrowed indices.
        if stats.null_count() as f64 / len as f64 > 0.9 {
            let compressed_bytes = value_count as usize * (elem_bytes + idx_width);
            if compressed_bytes >= uncompressed_bytes {
                return CompressionEstimate::Verdict(EstimateVerdict::Skip);
            }
            return CompressionEstimate::Verdict(EstimateVerdict::Ratio(
                uncompressed_bytes as f64 / compressed_bytes as f64,
            ));
        }

        let (_, most_frequent_count) = stats
            .erased()
            .most_frequent_value_and_count()
            .vortex_expect(
                "this must be present since `SparseScheme` declared that we need distinct values",
            );

        // If the most frequent value is the only value, we should compress as constant instead.
        if most_frequent_count == value_count {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        debug_assert!(value_count > most_frequent_count);

        // See if the most frequent value accounts for >= 90% of the set values.
        let freq = most_frequent_count as f64 / value_count as f64;
        if freq < 0.9 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // The fill value is not null, so `Sparse::encode` stores every non-fill value and every
        // null as a patch with a narrowed index and (when nulls exist) a patch validity bitmap.
        let exception_count = len - most_frequent_count as usize;
        let patch_validity = if has_nulls {
            exception_count.div_ceil(8)
        } else {
            0
        };
        let compressed_bytes = exception_count * (elem_bytes + idx_width) + patch_validity;
        if compressed_bytes == 0 || compressed_bytes >= uncompressed_bytes {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        CompressionEstimate::Verdict(EstimateVerdict::Ratio(
            uncompressed_bytes as f64 / compressed_bytes as f64,
        ))
    }

    /// Children: values=0, indices=1. The values carry the validity of the patched rows.
    fn encode(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Encoded> {
        let len = data.array_len();
        let stats = data.integer_stats(exec_ctx);
        let array = data.array();

        let (most_frequent_value, most_frequent_count) = stats
            .erased()
            .most_frequent_value_and_count()
            .vortex_expect(
                "this must be present since `SparseScheme` declared that we need distinct values",
            );
        let fill = Scalar::primitive_value(
            most_frequent_value,
            most_frequent_value.ptype(),
            array.dtype().nullability(),
        );

        if most_frequent_count as usize == len {
            // If the most frequent value is the only value, we should compress as constant instead.
            return Ok(Encoded::leaf(ConstantArray::new(fill, len).into_array()));
        }

        let sparse_encoded = Sparse::encode(array, Some(fill), exec_ctx)?;
        let Some(sparse) = sparse_encoded.as_opt::<Sparse>() else {
            return Ok(Encoded::leaf(sparse_encoded));
        };

        let values = sparse
            .patches()
            .values()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?;
        let indices = sparse
            .patches()
            .indices()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?;

        let fill = sparse.fill_scalar().clone();
        Ok(Encoded::new(
            vec![
                Child::new(self, 0, values.into_array()),
                Child::new(self, 1, indices.into_array()),
            ],
            Vec::new(),
            move |children, exec_ctx| {
                let [values, indices] = children.compressed()?;
                Sparse::try_new(indices, values, len, fill, exec_ctx).map(IntoArray::into_array)
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
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_error::VortexResult;
    use vortex_fastlanes::BitPacked;

    use crate::BtrBlocksCompressorBuilder;
    use crate::SESSION;
    use crate::schemes::integer::BITPACKING_V1;
    use crate::schemes::integer::SparseScheme;

    #[test]
    fn test_integer_sparse_prices_index_bytes_so_bitpacking_wins_on_u8() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // 10,000 u8 values (90% 0, 10% 1): 1,000 patches cost 1,000 * (1 + 2) = 3,000 B (~3.33x),
        // whereas 1-bit BitPacking costs ~1,280 B (~7.8x). Without pricing patch indices, Sparse
        // claimed 10.0x and beat BitPacking while writing ~40% more bytes.
        let input =
            PrimitiveArray::from_iter((0..10_000).map(|i| u8::from(i % 10 == 0))).into_array();

        let compressor = BtrBlocksCompressorBuilder::empty()
            .with_new_scheme(&BITPACKING_V1)
            .with_new_scheme(&SparseScheme)
            .build();
        let compressed = compressor.compress(&input, &mut ctx)?;
        assert!(
            compressed.is::<BitPacked>(),
            "expected BitPacked over Sparse, got {}",
            compressed.encoding_id()
        );
        assert_arrays_eq!(compressed, input, &mut ctx);
        Ok(())
    }
}
