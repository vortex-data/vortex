// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! String compression schemes.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::varbin::VarBinArrayExt;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_error::VortexResult;
use vortex_fsst::FSST;
use vortex_fsst::FSSTArrayExt;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_sparse::Sparse;

use super::integer::IntDictScheme;
use super::integer::SparseScheme as IntSparseScheme;
use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

/// FSST (Fast Static Symbol Table) compression.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FSSTScheme;

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
pub use vortex_compressor::builtins::is_utf8_string;
pub use vortex_compressor::stats::StringStats;

impl Scheme for FSSTScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.string.fsst"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_utf8_string(canonical)
    }

    /// Children: lengths=0, code_offsets=1.
    fn num_children(&self) -> usize {
        2
    }

    fn expected_compression_ratio(
        &self,
        _data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> CompressionEstimate {
        CompressionEstimate::Sample
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let utf8 = data.array_as_utf8();
        let compressor_fsst = fsst_train_compressor(&utf8);
        let fsst = fsst_compress(&utf8, utf8.len(), utf8.dtype(), &compressor_fsst);

        let compressed_original_lengths = compressor.compress_child(
            &fsst
                .uncompressed_lengths()
                .to_primitive()
                .narrow()?
                .into_array(),
            &ctx,
            self.id(),
            0,
        )?;

        let compressed_codes_offsets = compressor.compress_child(
            &fsst.codes().offsets().to_primitive().narrow()?.into_array(),
            &ctx,
            self.id(),
            1,
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
        )?;

        Ok(fsst.into_array())
    }
}

impl Scheme for NullDominatedSparseScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.string.sparse"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_utf8_string(canonical)
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
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> CompressionEstimate {
        let len = data.array_len() as f64;
        let stats = data.string_stats();
        let value_count = stats.value_count();

        // All-null arrays should be compressed as constant instead anyways.
        if value_count == 0 {
            return CompressionEstimate::Skip;
        }

        // If the majority (90%) of values is null, this will compress well.
        if stats.null_count() as f64 / len > 0.9 {
            return CompressionEstimate::Ratio(len / value_count as f64);
        }

        // Otherwise we don't go this route.
        CompressionEstimate::Skip
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        // We pass None as we only run this pathway for NULL-dominated string arrays.
        let sparse_encoded = Sparse::encode(data.array(), None)?;

        if let Some(sparse) = sparse_encoded.as_opt::<Sparse>() {
            // Compress the indices only (not the values for strings).
            let indices = sparse.patches().indices().to_primitive().narrow()?;
            let compressed_indices =
                compressor.compress_child(&indices.into_array(), &ctx, self.id(), 0)?;

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
        is_utf8_string(canonical)
    }

    fn expected_compression_ratio(
        &self,
        _data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> CompressionEstimate {
        CompressionEstimate::Sample
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let compacted = data.array_as_utf8().compact_buffers()?;
        Ok(vortex_zstd::Zstd::from_var_bin_view_without_dict(&compacted, 3, 8192)?.into_array())
    }
}

#[cfg(all(feature = "zstd", feature = "unstable_encodings"))]
impl Scheme for ZstdBuffersScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.string.zstd_buffers"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_utf8_string(canonical)
    }

    fn expected_compression_ratio(
        &self,
        _data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> CompressionEstimate {
        CompressionEstimate::Sample
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        Ok(
            vortex_zstd::ZstdBuffers::compress(data.array(), 3, &vortex_array::LEGACY_SESSION)?
                .into_array(),
        )
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::builders::ArrayBuilder;
    use vortex_array::builders::VarBinViewBuilder;
    use vortex_array::display::DisplayOptions;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_error::VortexResult;

    use crate::BtrBlocksCompressor;

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
        let compressed = btr.compress(&array_ref)?;
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
        let compressed = btr.compress(&array_ref)?;
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
    use vortex_array::IntoArray;
    use vortex_array::arrays::Constant;
    use vortex_array::arrays::Dict;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_error::VortexResult;
    use vortex_fsst::FSST;

    use crate::BtrBlocksCompressor;

    #[test]
    fn test_constant_compressed() -> VortexResult<()> {
        let strings: Vec<Option<&str>> = vec![Some("constant_value"); 100];
        let array = VarBinViewArray::from_iter(strings, DType::Utf8(Nullability::NonNullable));
        let array_ref = array.into_array();
        let compressed = BtrBlocksCompressor::default().compress(&array_ref)?;
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
        let compressed = BtrBlocksCompressor::default().compress(&array_ref)?;
        assert!(compressed.is::<Dict>());
        Ok(())
    }

    #[test]
    fn test_fsst_compressed() -> VortexResult<()> {
        let mut strings = Vec::with_capacity(1000);
        for i in 0..1000 {
            strings.push(Some(format!(
                "this_is_a_common_prefix_with_some_variation_{i}_and_a_common_suffix_pattern"
            )));
        }
        let array = VarBinViewArray::from_iter(strings, DType::Utf8(Nullability::NonNullable));
        let array_ref = array.into_array();
        let compressed = BtrBlocksCompressor::default().compress(&array_ref)?;
        assert!(compressed.is::<FSST>());
        Ok(())
    }
}
