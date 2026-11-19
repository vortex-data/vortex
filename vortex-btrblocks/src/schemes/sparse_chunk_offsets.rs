// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sparse scheme variants that also write patch chunk offsets.
//!
//! Sparse arrays with chunk offsets serialize under the experimental [`sparse_v2_id`], so each
//! sparse scheme writes them only through [`WithChunkOffsets`], which its
//! [`refine`](Scheme::refine) picks when that ID is allowed.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::scalar::Scalar;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_error::VortexResult;
use vortex_sparse::Sparse;
use vortex_sparse::sparse_v2_id;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::GenerateStatsOptions;
use crate::Scheme;
use crate::SchemeExt;

/// A sparse scheme whose compression can optionally keep patch chunk offsets.
pub(crate) trait ChunkOffsetsScheme: Scheme {
    /// Compresses like [`Scheme::compress`], keeping the chunk offsets if `with_chunk_offsets`.
    ///
    /// The chunk offsets become child [`Scheme::num_children`], after the scheme's own children.
    fn compress_sparse(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
        with_chunk_offsets: bool,
    ) -> VortexResult<ArrayRef>;
}

/// Returns the variant of `scheme` to use given which serialized IDs are `allowed`.
pub(crate) fn refine_sparse<'a, S: ChunkOffsetsScheme>(
    scheme: &'a S,
    with_chunk_offsets: &'a WithChunkOffsets<S>,
    allowed: &dyn Fn(&ArrayId) -> bool,
) -> &'a dyn Scheme {
    if allowed(&sparse_v2_id()) {
        with_chunk_offsets
    } else {
        scheme
    }
}

/// The variant of sparse scheme `S` that writes patch chunk offsets.
///
/// It shares the scheme ID of `S` and behaves like it, except that it also writes
/// [`sparse_v2_id`] and has one more child, the chunk offsets.
#[derive(Debug)]
pub(crate) struct WithChunkOffsets<S>(pub(crate) S);

impl<S: ChunkOffsetsScheme> Scheme for WithChunkOffsets<S> {
    fn scheme_name(&self) -> &'static str {
        self.0.scheme_name()
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        self.0.matches(canonical)
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        let mut encodings = self.0.produced_encodings();
        encodings.push(sparse_v2_id());
        encodings
    }

    fn stats_options(&self) -> GenerateStatsOptions {
        self.0.stats_options()
    }

    fn num_children(&self) -> usize {
        self.0.num_children() + 1
    }

    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        self.0.descendant_exclusions()
    }

    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        self.0.ancestor_exclusions()
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        self.0
            .expected_compression_ratio(data, compress_ctx, exec_ctx)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        self.0
            .compress_sparse(compressor, data, compress_ctx, exec_ctx, true)
    }
}

/// Encodes `array` as sparse, with patch chunk offsets if `with_chunk_offsets`.
pub(crate) fn encode_sparse(
    array: &ArrayRef,
    fill_value: Option<Scalar>,
    with_chunk_offsets: bool,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if with_chunk_offsets {
        Sparse::encode_with_chunk_offsets(array, fill_value, exec_ctx)
    } else {
        Sparse::encode(array, fill_value, exec_ctx)
    }
}

/// Narrows and compresses `chunk_offsets`, if any, as the child after the children of `scheme`.
///
/// Sparse arrays are often tiny, so the offsets, one per 1024 rows, can dominate their size
/// unless they are compressed like the indices.
pub(crate) fn compress_chunk_offsets<S: Scheme + ?Sized>(
    compressor: &CascadingCompressor,
    chunk_offsets: Option<&ArrayRef>,
    compress_ctx: &CompressorContext,
    scheme: &S,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    chunk_offsets
        .map(|offsets| {
            let narrowed = offsets
                .clone()
                .execute::<PrimitiveArray>(exec_ctx)?
                .narrow(exec_ctx)?
                .into_array();
            compressor.compress_child(
                &narrowed,
                compress_ctx,
                scheme.id(),
                scheme.num_children(),
                exec_ctx,
            )
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayId;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_error::VortexResult;
    use vortex_sparse::Sparse;
    use vortex_sparse::SparseExt as _;
    use vortex_sparse::sparse_v1_id;
    use vortex_sparse::sparse_v2_id;

    use crate::BtrBlocksCompressorBuilder;
    use crate::SESSION;
    use crate::Scheme;
    use crate::SchemeExt;
    use crate::schemes::boolean::BoolSparseScheme;
    use crate::schemes::fixed_size_list::FixedSizeListSparseScheme;
    use crate::schemes::float;
    use crate::schemes::integer;
    use crate::schemes::string;

    /// Each sparse scheme refines to its chunk offsets variant exactly when v2 is allowed.
    #[rstest]
    fn refine_writes_chunk_offsets_when_v2_id_is_allowed(
        #[values(false, true)] allow_v2: bool,
        #[values(
            &integer::SparseScheme as &dyn Scheme,
            &float::NullDominatedSparseScheme,
            &string::NullDominatedSparseScheme,
            &BoolSparseScheme,
            &FixedSizeListSparseScheme
        )]
        scheme: &'static dyn Scheme,
    ) {
        let allowed = |id: &ArrayId| *id == sparse_v1_id() || (allow_v2 && *id == sparse_v2_id());
        let refined = scheme.refine(&allowed);
        assert_eq!(refined.id(), scheme.id());

        let mut expected = scheme.produced_encodings();
        if allow_v2 {
            expected.push(sparse_v2_id());
        }
        assert_eq!(refined.produced_encodings(), expected);
        assert_eq!(
            refined.num_children(),
            scheme.num_children() + usize::from(allow_v2)
        );
    }

    #[test]
    fn compress_with_v2_allowed_writes_chunk_offsets() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let input = PrimitiveArray::from_option_iter(
            (0..100_000i64).map(|i| (i % 997 == 0).then_some(i * 31 % 1000)),
        )
        .into_array();
        let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
            .unrestricted()
            .build();

        let compressed = compressor.compress(&input, &mut ctx)?;

        let sparse = compressed.as_::<Sparse>();
        let chunk_offsets = sparse
            .patches()
            .chunk_offsets()
            .clone()
            .expect("v2 sparse arrays carry chunk offsets");
        assert_eq!(chunk_offsets.len(), input.len().div_ceil(1024));
        assert_arrays_eq!(compressed, input, &mut ctx);
        Ok(())
    }
}
