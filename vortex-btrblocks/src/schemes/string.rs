// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! String compression schemes.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::varbin::VarBinArrayExt;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::DeferredEstimate;
use vortex_compressor::estimate::EstimateVerdict;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_error::VortexResult;
use vortex_fsst::FSST;
use vortex_fsst::FSSTArrayExt;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_sparse::Sparse;
use vortex_sparse::SparseExt as _;

use super::integer::IntDictScheme;
use super::integer::SparseScheme as IntSparseScheme;
use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

/// FSST (Fast Static Symbol Table) compression.
///
/// One of the two string-fragmentation schemes in the default
/// [`crate::ALL_SCHEMES`] (alongside `OnPairScheme`); the sample-based selector
/// keeps whichever is smaller per column. FSST compresses faster, OnPair
/// usually wins on ratio.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FSSTScheme;

#[cfg(feature = "unstable_encodings")]
pub use onpair::OnPairScheme;

/// Sparse encoding for null-dominated arrays.
///
/// This is the same as the integer `SparseScheme`, but we only use this for null-dominated arrays.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NullDominatedSparseScheme;

/// Zstd compression without dictionaries (nvCOMP compatible).
#[cfg(feature = "zstd")]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ZstdScheme;

/// Zstd buffer-level compression preserving array layout for GPU decompression.
#[cfg(all(feature = "zstd", feature = "unstable_encodings"))]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ZstdBuffersScheme;

// Re-export builtin schemes from vortex-compressor.
pub use vortex_compressor::builtins::StringConstantScheme;
pub use vortex_compressor::builtins::StringDictScheme;
pub use vortex_compressor::stats::StringStats;

impl Scheme for FSSTScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.string.fsst"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_utf8()
    }

    /// Children: lengths=0, code_offsets=1.
    fn num_children(&self) -> usize {
        2
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let utf8 = data.array_as_varbinview().into_owned();
        let compressor_fsst = fsst_train_compressor(&utf8);
        let fsst = fsst_compress(&utf8, utf8.len(), utf8.dtype(), &compressor_fsst, exec_ctx);

        let uncompressed_lengths_primitive = fsst
            .uncompressed_lengths()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?;
        let compressed_original_lengths = compressor.compress_child(
            &uncompressed_lengths_primitive.into_array(),
            &compress_ctx,
            self.id(),
            0,
            exec_ctx,
        )?;

        let codes_offsets_primitive = fsst
            .codes()
            .offsets()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?;
        let compressed_codes_offsets = compressor.compress_child(
            &codes_offsets_primitive.into_array(),
            &compress_ctx,
            self.id(),
            1,
            exec_ctx,
        )?;
        let compressed_codes = VarBinArray::try_new(
            compressed_codes_offsets,
            fsst.codes().bytes().clone(),
            fsst.codes().dtype().clone(),
            fsst.codes().validity()?,
        )?;

        let fsst = FSST::try_new(
            fsst.dtype().clone(),
            fsst.symbols().clone(),
            fsst.symbol_lengths().clone(),
            compressed_codes,
            compressed_original_lengths,
            exec_ctx,
        )?;

        Ok(fsst.into_array())
    }
}

impl Scheme for NullDominatedSparseScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.string.sparse"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_utf8()
    }

    /// Children: indices=0.
    fn num_children(&self) -> usize {
        1
    }

    /// The indices of a null-dominated sparse array should not be sparse-encoded again.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![
            DescendantExclusion {
                excluded: IntSparseScheme.id(),
                children: ChildSelection::All,
            },
            DescendantExclusion {
                excluded: IntDictScheme.id(),
                children: ChildSelection::All,
            },
        ]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let len = data.array_len() as f64;
        let stats = data.varbinview_stats(exec_ctx);
        let value_count = stats.value_count();

        // All-null arrays should be compressed as constant instead anyways.
        if value_count == 0 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // If the majority (90%) of values is null, this will compress well.
        if stats.null_count() as f64 / len > 0.9 {
            return CompressionEstimate::Verdict(EstimateVerdict::Ratio(len / value_count as f64));
        }

        // Otherwise we don't go this route.
        CompressionEstimate::Verdict(EstimateVerdict::Skip)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        // We pass None as we only run this pathway for NULL-dominated string arrays.
        let sparse_encoded = Sparse::encode(data.array(), None, exec_ctx)?;

        if let Some(sparse) = sparse_encoded.as_opt::<Sparse>() {
            // Compress the indices only (not the values for strings).
            let indices = sparse
                .patches()
                .indices()
                .clone()
                .execute::<PrimitiveArray>(exec_ctx)?
                .narrow(exec_ctx)?;
            let compressed_indices = compressor.compress_child(
                &indices.into_array(),
                &compress_ctx,
                self.id(),
                0,
                exec_ctx,
            )?;

            Sparse::try_new(
                compressed_indices,
                sparse.patches().values().clone(),
                sparse.len(),
                sparse.fill_scalar().clone(),
            )
            .map(|a| a.into_array())
        } else {
            Ok(sparse_encoded)
        }
    }
}

#[cfg(feature = "zstd")]
impl Scheme for ZstdScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.string.zstd"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_utf8()
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let compacted = data.array_as_varbinview().into_owned().compact_buffers()?;
        Ok(
            vortex_zstd::Zstd::from_var_bin_view_without_dict(&compacted, 3, 8192, exec_ctx)?
                .into_array(),
        )
    }
}

#[cfg(all(feature = "zstd", feature = "unstable_encodings"))]
impl Scheme for ZstdBuffersScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.string.zstd_buffers"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_utf8()
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        Ok(vortex_zstd::ZstdBuffers::compress(data.array(), 3, exec_ctx.session())?.into_array())
    }
}

#[cfg(feature = "unstable_encodings")]
mod onpair {
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::primitive::PrimitiveArrayExt;
    use vortex_compressor::estimate::CompressionEstimate;
    use vortex_compressor::estimate::DeferredEstimate;
    use vortex_compressor::scheme::SchemeId;
    use vortex_error::VortexResult;
    use vortex_onpair::DEFAULT_DICT12_CONFIG;
    use vortex_onpair::OnPair;
    use vortex_onpair::OnPairArrayExt;
    use vortex_onpair::OnPairArraySlotsExt;
    use vortex_onpair::onpair_compress;

    use crate::ArrayAndStats;
    use crate::CascadingCompressor;
    use crate::CompressorContext;
    use crate::Scheme;
    use crate::SchemeExt;
    use crate::schemes::integer::try_compress_delta;

    /// OnPair short-string compression (dict-12).
    ///
    /// A default string-fragmentation scheme (alongside [`super::FSSTScheme`]) —
    /// targets large columns of short-to-medium strings with high lexical
    /// overlap, like URLs or log lines. Uses a learned dictionary of frequent
    /// adjacent substrings (built by the OnPair trainer at compress time) and
    /// 12-bit token codes stored as a u16 child, with offsets /
    /// uncompressed-lengths flowing through the cascading compressor like any
    /// other primitive children.
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub struct OnPairScheme;

    impl Scheme for OnPairScheme {
        fn scheme_name(&self) -> &'static str {
            "vortex.string.onpair"
        }

        fn matches(&self, canonical: &Canonical) -> bool {
            canonical.dtype().is_utf8()
        }

        /// 4 primitive slot children flow through the cascading compressor:
        /// `dict_offsets` (u32 → typically `FoR`/`BitPacked`), `codes` (u16 →
        /// `FastLanes::BitPacked` to exactly `bits` = 12 by default),
        /// `codes_offsets` (u32 → `FoR`), `uncompressed_lengths` (i32 → narrow
        /// + `FoR`). Validity stays untouched.
        fn num_children(&self) -> usize {
            4
        }

        fn expected_compression_ratio(
            &self,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> CompressionEstimate {
            CompressionEstimate::Deferred(DeferredEstimate::Sample)
        }

        fn compress(
            &self,
            compressor: &CascadingCompressor,
            data: &ArrayAndStats,
            compress_ctx: CompressorContext,
            exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            let utf8 = data.array_as_varbinview().into_owned();
            let onpair_array =
                onpair_compress(&utf8, utf8.len(), utf8.dtype(), DEFAULT_DICT12_CONFIG)?;

            let dict_offsets = compress_offsets_child(
                compressor,
                onpair_array.dict_offsets(),
                &compress_ctx,
                self.id(),
                0,
                exec_ctx,
            )?;
            let codes = compress_primitive_child(
                compressor,
                onpair_array.codes(),
                &compress_ctx,
                self.id(),
                1,
                exec_ctx,
            )?;
            let codes_offsets = compress_offsets_child(
                compressor,
                onpair_array.codes_offsets(),
                &compress_ctx,
                self.id(),
                2,
                exec_ctx,
            )?;
            let uncompressed_lengths = compress_primitive_child(
                compressor,
                onpair_array.uncompressed_lengths(),
                &compress_ctx,
                self.id(),
                3,
                exec_ctx,
            )?;

            Ok(OnPair::try_new(
                onpair_array.dtype().clone(),
                onpair_array.dict_bytes_handle().clone(),
                dict_offsets,
                codes,
                codes_offsets,
                uncompressed_lengths,
                onpair_array.array_validity(),
                onpair_array.bits(),
            )?
            .into_array())
        }
    }

    /// Narrow a primitive child to its tightest int type, then forward it to
    /// the cascading compressor.
    fn compress_primitive_child(
        compressor: &CascadingCompressor,
        child: &ArrayRef,
        compress_ctx: &CompressorContext,
        scheme_id: SchemeId,
        child_idx: usize,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let narrowed = child
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?
            .into_array();
        compressor.compress_child(&narrowed, compress_ctx, scheme_id, child_idx, exec_ctx)
    }

    /// Minimum child length before delta is even attempted. Delta carries fixed
    /// overhead (a separate `bases` array plus FastLanes' 1024-element lane
    /// packing), so on short children it can only lose.
    const OFFSETS_DELTA_MIN_LEN: usize = 2048;

    /// Compress a monotonic offsets child. For children of at least
    /// [`OFFSETS_DELTA_MIN_LEN`] it tries both the normal cascading path and a
    /// delta path and keeps whichever produces fewer bytes; shorter children
    /// skip delta entirely. `dict_offsets` and `codes_offsets` are cumulative
    /// (monotonic), so delta (per-entry deltas) usually packs much tighter than
    /// FoR+bitpacking over the full range.
    fn compress_offsets_child(
        compressor: &CascadingCompressor,
        child: &ArrayRef,
        compress_ctx: &CompressorContext,
        scheme_id: SchemeId,
        child_idx: usize,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let narrowed = child
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?
            .into_array();
        let plain =
            compressor.compress_child(&narrowed, compress_ctx, scheme_id, child_idx, exec_ctx)?;
        if narrowed.len() < OFFSETS_DELTA_MIN_LEN {
            return Ok(plain);
        }
        let delta = try_compress_delta(
            compressor,
            &narrowed,
            compress_ctx,
            scheme_id,
            child_idx,
            exec_ctx,
        )?;
        if delta.nbytes() < plain.nbytes() {
            Ok(delta)
        } else {
            Ok(plain)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::builders::ArrayBuilder;
    use vortex_array::builders::VarBinViewBuilder;
    use vortex_array::display::DisplayOptions;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::BtrBlocksCompressor;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[test]
    fn test_strings() -> VortexResult<()> {
        let mut strings = Vec::new();
        for _ in 0..1024 {
            strings.push(Some("hello-world-1234"));
        }
        for _ in 0..1024 {
            strings.push(Some("hello-world-56789"));
        }
        let strings = VarBinViewArray::from_iter(strings, DType::Utf8(Nullability::NonNullable));

        let array_ref = strings.into_array();
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array_ref, &mut SESSION.create_execution_ctx())?;
        assert_eq!(compressed.len(), 2048);

        let display = compressed
            .display_as(DisplayOptions::MetadataOnly)
            .to_string()
            .to_lowercase();
        assert_eq!(display, "vortex.dict(utf8, len=2048)");

        Ok(())
    }

    #[test]
    fn test_sparse_nulls() -> VortexResult<()> {
        let mut strings = VarBinViewBuilder::with_capacity(DType::Utf8(Nullability::Nullable), 100);
        strings.append_nulls(99);

        strings.append_value("one little string");

        let strings = strings.finish_into_varbinview();

        let array_ref = strings.into_array();
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array_ref, &mut SESSION.create_execution_ctx())?;
        assert_eq!(compressed.len(), 100);

        let display = compressed
            .display_as(DisplayOptions::MetadataOnly)
            .to_string()
            .to_lowercase();
        assert_eq!(display, "vortex.sparse(utf8?, len=100)");

        Ok(())
    }
}

/// Tests to verify that each string compression scheme produces the expected encoding.
#[cfg(test)]
mod scheme_selection_tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Constant;
    use vortex_array::arrays::Dict;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexResult;
    use vortex_fsst::FSST;
    use vortex_session::VortexSession;

    use crate::BtrBlocksCompressor;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[test]
    fn test_constant_compressed() -> VortexResult<()> {
        let strings: Vec<Option<&str>> = vec![Some("constant_value"); 100];
        let array = VarBinViewArray::from_iter(strings, DType::Utf8(Nullability::NonNullable));
        let array_ref = array.into_array();
        let compressed = BtrBlocksCompressor::default()
            .compress(&array_ref, &mut SESSION.create_execution_ctx())?;
        assert!(compressed.is::<Constant>());
        Ok(())
    }

    #[test]
    fn test_dict_compressed() -> VortexResult<()> {
        let distinct_values = ["apple", "banana", "cherry"];
        let mut strings = Vec::with_capacity(1000);
        for i in 0..1000 {
            strings.push(Some(distinct_values[i % 3]));
        }
        let array = VarBinViewArray::from_iter(strings, DType::Utf8(Nullability::NonNullable));
        let array_ref = array.into_array();
        let compressed = BtrBlocksCompressor::default()
            .compress(&array_ref, &mut SESSION.create_execution_ctx())?;
        assert!(compressed.is::<Dict>());
        Ok(())
    }

    #[cfg(feature = "unstable_encodings")]
    #[test]
    fn test_onpair_in_default_scheme_list() {
        use crate::SchemeExt;
        use crate::schemes::string::OnPairScheme;

        let ids: Vec<_> = crate::ALL_SCHEMES.iter().map(|s| s.id()).collect();
        assert!(
            ids.contains(&OnPairScheme.id()),
            "OnPairScheme not registered in ALL_SCHEMES"
        );
    }

    #[cfg(feature = "unstable_encodings")]
    #[test]
    fn test_onpair_compressed() -> VortexResult<()> {
        // Dictionary-style string corpus: high lexical overlap, short rows.
        // OnPair beats FSST on this corpus, so it wins the sample-based
        // comparison even though both are registered by default.
        let mut strings = Vec::with_capacity(1000);
        for i in 0..1000 {
            strings.push(Some(format!(
                "this_is_a_common_prefix_with_some_variation_{i}_and_a_common_suffix_pattern"
            )));
        }
        let array = VarBinViewArray::from_iter(strings, DType::Utf8(Nullability::NonNullable));
        let array_ref = array.into_array();
        let compressed = BtrBlocksCompressor::default()
            .compress(&array_ref, &mut SESSION.create_execution_ctx())?;
        assert!(
            compressed.is::<vortex_onpair::OnPair>(),
            "expected OnPair, got {}",
            compressed.encoding_id()
        );
        Ok(())
    }

    /// FSST is registered in the default scheme list (alongside OnPair), and an
    /// FSST-only builder still produces an FSST array.
    #[test]
    fn test_fsst_in_default_scheme_list() -> VortexResult<()> {
        use crate::BtrBlocksCompressorBuilder;
        use crate::SchemeExt;
        use crate::schemes::string::FSSTScheme;

        // FSST is registered by default.
        assert!(
            crate::ALL_SCHEMES.iter().any(|s| s.id() == FSSTScheme.id()),
            "FSSTScheme should be in ALL_SCHEMES",
        );

        // An FSST-only builder still produces an FSST array for FSST-favourable
        // input.
        let mut strings = Vec::with_capacity(1000);
        for i in 0..1000 {
            strings.push(Some(format!(
                "this_is_a_common_prefix_with_some_variation_{i}_and_a_common_suffix_pattern"
            )));
        }
        let array = VarBinViewArray::from_iter(strings, DType::Utf8(Nullability::NonNullable));
        let array_ref = array.into_array();

        let compressor = BtrBlocksCompressorBuilder::empty()
            .with_new_scheme(&FSSTScheme)
            .build();
        let compressed = compressor.compress(&array_ref, &mut SESSION.create_execution_ctx())?;
        assert!(
            compressed.is::<FSST>(),
            "expected FSST when only FSSTScheme is registered, got {}",
            compressed.encoding_id()
        );
        Ok(())
    }
}
