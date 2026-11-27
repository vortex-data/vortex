// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sparse encoding for null-dominated float arrays.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::PType;
use vortex_compressor::builtins::IntDictScheme;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_sparse::Sparse;
use vortex_sparse::SparseExt as _;

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

/// Sparse encoding for null-dominated float arrays.
///
/// This is the same as the integer `SparseScheme`, but we only use this for null-dominated arrays.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NullDominatedSparseScheme;

impl Scheme for NullDominatedSparseScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.sparse"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_float()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Sparse.id()]
    }

    /// Children: values=0, indices=1.
    fn num_children(&self) -> usize {
        2
    }

    /// Patch values (child 0) are already non-null, so null-dominated Sparse is pointless on them.
    /// Sparse indices (child 1) are strictly increasing unique positions, so Dict, RunEnd, RLE,
    /// and Sparse are all pointless on them.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![
            DescendantExclusion {
                excluded: NullDominatedSparseScheme.id(),
                children: ChildSelection::One(0),
            },
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
                excluded: IntSparseScheme.id(),
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
        let stats = data.float_stats(exec_ctx);
        let value_count = stats.value_count() as usize;

        // All-null arrays should be compressed as constant instead anyways.
        if value_count == 0 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // Only null-dominated arrays (> 90% null) go this route.
        if stats.null_count() as f64 / len as f64 <= 0.9 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        let elem_bytes = data.array_as_primitive().ptype().byte_width();
        let validity_bytes = len.div_ceil(8);
        let uncompressed_bytes = len * elem_bytes + validity_bytes;
        let idx_width =
            PType::min_unsigned_ptype_for_value(len.saturating_sub(1) as u64).byte_width();
        let compressed_bytes = value_count * (elem_bytes + idx_width);
        if compressed_bytes >= uncompressed_bytes {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        CompressionEstimate::Verdict(EstimateVerdict::Ratio(
            uncompressed_bytes as f64 / compressed_bytes as f64,
        ))
    }

    /// Children: values=0, indices=1. The values carry no nulls, so the indices define validity
    /// and the float schemes can compress the values directly.
    fn encode(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Encoded> {
        // We pass None as we only run this pathway for NULL-dominated float arrays.
        let sparse_encoded = Sparse::encode(data.array(), None, exec_ctx)?;
        let Some(sparse) = sparse_encoded.as_opt::<Sparse>() else {
            return Ok(Encoded::leaf(sparse_encoded));
        };

        let values = sparse.patches().values().clone();
        let indices = sparse
            .patches()
            .indices()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?;
        let len = sparse.len();
        let fill = sparse.fill_scalar().clone();
        Ok(Encoded::new(
            vec![
                Child::new(self, 0, values),
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
    use vortex_compressor::builtins::IntDictScheme;
    use vortex_compressor::scheme::ChildSelection;
    use vortex_error::VortexResult;
    use vortex_sparse::Sparse;
    use vortex_sparse::SparseExt as _;

    use super::NullDominatedSparseScheme;
    use crate::BtrBlocksCompressorBuilder;
    use crate::SESSION;
    use crate::Scheme;
    use crate::SchemeExt;
    use crate::schemes::integer::IntRLEScheme;
    use crate::schemes::integer::RunEndScheme;
    use crate::schemes::integer::SparseScheme as IntSparseScheme;

    #[test]
    fn test_float_sparse_compresses_patch_values() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // 95%-null f64 array with 3,277 distinct ALP-compressible decimal values (~26 KB raw).
        // Sparse beats dense ALP because 95% of rows are null, and compressing child 0 (values)
        // shrinks the patch values well below raw f64 size.
        let len = 65_536;
        let input = PrimitiveArray::from_option_iter((0..len).map(|i| {
            (i % 20 == 0).then_some((i / 20) as f64 * 0.25 + ((i * 7) % 13) as f64 * 0.01)
        }))
        .into_array();

        let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
            .unrestricted()
            .build();
        let compressed = compressor.compress(&input, &mut ctx)?;
        let sparse = compressed
            .as_opt::<Sparse>()
            .expect("expected Sparse for 95%-null float array");
        let patches = sparse.patches();
        let patch_values = patches.values();
        assert!(
            !patch_values.is_canonical(),
            "expected compressed patch values, got {}",
            patch_values.encoding_id()
        );
        assert!(patch_values.nbytes() < (patch_values.len() * size_of::<f64>() / 2) as u64);
        assert_arrays_eq!(compressed, input, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_float_sparse_prices_indices_against_uncompressed_bytes() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // 92%-null f32 array of 100,000 rows: 8,000 patches need u32 indices (4 B) + f32 values
        // (4 B) = 64,000 B vs 412,500 B uncompressed (~6.45x). With only two distinct valid values,
        // FloatDict (codes ~1 bit + 12,500 B validity) should beat Sparse when indices are priced.
        let len = 100_000;
        let input =
            PrimitiveArray::from_option_iter((0..len).map(|i| {
                (i % 13 == 0).then_some(if (i / 13) % 2 == 0 { 1.25f32 } else { 2.5f32 })
            }))
            .into_array();

        let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
            .unrestricted()
            .build();
        let compressed = compressor.compress(&input, &mut ctx)?;
        assert!(
            !compressed.is::<Sparse>(),
            "expected FloatDict/ALP over Sparse when patch indices are priced, got {}",
            compressed.display_tree()
        );
        assert_arrays_eq!(compressed, input, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_float_sparse_descendant_exclusions() {
        let exclusions = NullDominatedSparseScheme.descendant_exclusions();
        assert!(exclusions.iter().any(|rule| {
            rule.excluded == NullDominatedSparseScheme.id()
                && matches!(rule.children, ChildSelection::One(0))
        }));
        for id in [
            IntDictScheme.id(),
            RunEndScheme.id(),
            IntRLEScheme.id(),
            IntSparseScheme.id(),
        ] {
            assert!(
                exclusions.iter().any(|rule| {
                    rule.excluded == id && matches!(rule.children, ChildSelection::One(1))
                }),
                "missing child-1 exclusion for {id}"
            );
        }
    }
}
