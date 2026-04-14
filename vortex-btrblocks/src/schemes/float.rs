// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Float compression schemes.

use vortex_alp::ALP;
use vortex_alp::ALPArrayExt;
use vortex_alp::ALPArraySlotsExt;
use vortex_alp::ALPRDArrayExt;
use vortex_alp::ALPRDArrayOwnedExt;
use vortex_alp::RDEncoder;
use vortex_alp::alp_encode;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::ToCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Patched;
use vortex_array::arrays::patched::USE_EXPERIMENTAL_PATCHES;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::PType;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::DeferredEstimate;
use vortex_compressor::estimate::EstimateVerdict;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_sparse::Sparse;

use super::integer::SparseScheme as IntSparseScheme;
use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::compress_patches;
use crate::schemes::rle_ancestor_exclusions;
use crate::schemes::rle_descendant_exclusions;

/// ALP (Adaptive Lossless floating-Point) encoding.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ALPScheme;

/// ALPRD (ALP with Real Double) encoding variant.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ALPRDScheme;

/// Sparse encoding for null-dominated float arrays.
///
/// This is the same as the integer `SparseScheme`, but we only use this for null-dominated arrays.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NullDominatedSparseScheme;

/// Pco (pcodec) compression for floats.
#[cfg(feature = "pco")]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct PcoScheme;

// Re-export builtin schemes from vortex-compressor.
pub use vortex_compressor::builtins::FloatConstantScheme;
pub use vortex_compressor::builtins::FloatDictScheme;
pub use vortex_compressor::builtins::is_float_primitive;
pub use vortex_compressor::stats::FloatStats;

/// RLE scheme for float arrays.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FloatRLEScheme;

impl Scheme for ALPScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.alp"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_float_primitive(canonical)
    }

    /// Children: encoded_ints=0.
    fn num_children(&self) -> usize {
        1
    }

    fn expected_compression_ratio(
        &self,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> CompressionEstimate {
        // ALP encodes floats as integers. Without integer compression afterward, the encoded ints
        // are the same size.
        if ctx.finished_cascading() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // We don't support ALP for f16.
        if data.array_as_primitive().ptype() == PType::F16 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let alp_encoded = alp_encode(
            data.array_as_primitive(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;

        // Compress the ALP ints.
        let compressed_alp_ints =
            compressor.compress_child(alp_encoded.encoded(), &ctx, self.id(), 0)?;

        let alp_stats = alp_encoded.as_array().statistics().to_owned();
        let exponents = alp_encoded.exponents();

        if *USE_EXPERIMENTAL_PATCHES {
            let patches = alp_encoded.patches();

            // Create ALP array without interior patches.
            let alp_array = ALP::new(compressed_alp_ints, exponents, None).into_array();

            match patches {
                None => Ok(alp_array),
                Some(p) => Ok(Patched::from_array_and_patches(
                    alp_array,
                    &p,
                    &mut LEGACY_SESSION.create_execution_ctx(),
                )?
                .with_stats_set(alp_stats)
                .into_array()),
            }
        } else {
            let patches = alp_encoded.patches().map(compress_patches).transpose()?;

            Ok(ALP::new(compressed_alp_ints, exponents, patches).into_array())
        }
    }
}

impl Scheme for ALPRDScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.alprd"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_float_primitive(canonical)
    }

    fn expected_compression_ratio(
        &self,
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> CompressionEstimate {
        // We don't support ALPRD for f16.
        if data.array_as_primitive().ptype() == PType::F16 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let primitive_array = data.array_as_primitive();

        let encoder = match primitive_array.ptype() {
            PType::F32 => RDEncoder::new(primitive_array.as_slice::<f32>()),
            PType::F64 => RDEncoder::new(primitive_array.as_slice::<f64>()),
            ptype => vortex_panic!("cannot ALPRD compress ptype {ptype}"),
        };

        let alp_rd = encoder.encode(primitive_array);
        let dtype = alp_rd.dtype().clone();
        let right_bit_width = alp_rd.right_bit_width();
        let mut parts = ALPRDArrayOwnedExt::into_data_parts(alp_rd);
        parts.left_parts_patches = parts.left_parts_patches.map(compress_patches).transpose()?;

        Ok(vortex_alp::ALPRD::try_new(
            dtype,
            parts.left_parts,
            parts.left_parts_dictionary,
            parts.right_parts,
            right_bit_width,
            parts.left_parts_patches,
        )?
        .into_array())
    }
}

impl Scheme for NullDominatedSparseScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.sparse"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_float_primitive(canonical)
    }

    /// Children: indices=0.
    fn num_children(&self) -> usize {
        1
    }

    /// The indices of a null-dominated sparse array should not be sparse-encoded again.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![DescendantExclusion {
            excluded: IntSparseScheme.id(),
            children: ChildSelection::All,
        }]
    }

    fn expected_compression_ratio(
        &self,
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> CompressionEstimate {
        let len = data.array_len() as f64;
        let stats = data.float_stats();
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
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        // We pass None as we only run this pathway for NULL-dominated float arrays.
        let sparse_encoded = Sparse::encode(data.array(), None)?;

        if let Some(sparse) = sparse_encoded.as_opt::<Sparse>() {
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

#[cfg(feature = "pco")]
impl Scheme for PcoScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.pco"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_float_primitive(canonical)
    }

    fn expected_compression_ratio(
        &self,
        _data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> CompressionEstimate {
        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        Ok(vortex_pco::Pco::from_primitive(
            data.array_as_primitive(),
            pco::DEFAULT_COMPRESSION_LEVEL,
            8192,
        )?
        .into_array())
    }
}

impl Scheme for FloatRLEScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.rle"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_float_primitive(canonical)
    }

    /// Children: values=0, indices=1, offsets=2.
    fn num_children(&self) -> usize {
        3
    }

    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        rle_descendant_exclusions()
    }

    fn ancestor_exclusions(&self) -> Vec<vortex_compressor::scheme::AncestorExclusion> {
        rle_ancestor_exclusions()
    }

    fn expected_compression_ratio(
        &self,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> CompressionEstimate {
        // RLE is only useful when we cascade it with another encoding.
        if ctx.finished_cascading() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        if data.float_stats().average_run_length() < super::integer::RUN_LENGTH_THRESHOLD {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        super::integer::rle_compress(self, compressor, data, ctx)
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builders::ArrayBuilder;
    use vortex_array::builders::PrimitiveBuilder;
    use vortex_array::display::DisplayOptions;
    use vortex_array::dtype::Nullability;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer_mut;
    use vortex_compressor::CascadingCompressor;
    use vortex_error::VortexResult;
    use vortex_fastlanes::RLE;

    use crate::BtrBlocksCompressor;
    use crate::schemes::float::FloatRLEScheme;

    #[test]
    fn test_empty() -> VortexResult<()> {
        let btr = BtrBlocksCompressor::default();
        let array = PrimitiveArray::new(Buffer::<f32>::empty(), Validity::NonNullable).into_array();
        let result = btr.compress(&array)?;

        assert!(result.is_empty());
        Ok(())
    }

    #[test]
    fn test_compress() -> VortexResult<()> {
        let mut values = buffer_mut![1.0f32; 1024];
        for i in 0..1024 {
            values[i] = (i % 50) as f32;
        }

        let array = values.into_array();
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array)?;
        assert_eq!(compressed.len(), 1024);

        let display = compressed
            .display_as(DisplayOptions::MetadataOnly)
            .to_string()
            .to_lowercase();
        assert_eq!(display, "vortex.dict(f32, len=1024)");

        Ok(())
    }

    #[test]
    fn test_rle_compression() -> VortexResult<()> {
        let mut values = Vec::new();
        values.extend(iter::repeat_n(1.5f32, 100));
        values.extend(iter::repeat_n(2.7f32, 200));
        values.extend(iter::repeat_n(3.15f32, 150));

        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);

        let compressor = CascadingCompressor::new(vec![&FloatRLEScheme]);
        let compressed = compressor.compress(&array.into_array())?;
        assert!(compressed.is::<RLE>());

        let expected = Buffer::copy_from(&values).into_array();
        assert_arrays_eq!(compressed, expected);
        Ok(())
    }

    #[test]
    fn test_sparse_compression() -> VortexResult<()> {
        let mut array = PrimitiveBuilder::<f32>::with_capacity(Nullability::Nullable, 100);
        array.append_value(f32::NAN);
        array.append_value(-f32::NAN);
        array.append_value(f32::INFINITY);
        array.append_value(-f32::INFINITY);
        array.append_value(0.0f32);
        array.append_value(-0.0f32);
        array.append_nulls(90);

        let array = array.finish_into_primitive().into_array();
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array)?;
        assert_eq!(compressed.len(), 96);

        let display = compressed
            .display_as(DisplayOptions::MetadataOnly)
            .to_string()
            .to_lowercase();
        assert_eq!(display, "vortex.sparse(f32?, len=96)");

        Ok(())
    }
}

/// Tests to verify that each float compression scheme produces the expected encoding.
#[cfg(test)]
mod scheme_selection_tests {
    use vortex_alp::ALP;
    use vortex_array::IntoArray;
    use vortex_array::arrays::Constant;
    use vortex_array::arrays::Dict;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::builders::ArrayBuilder;
    use vortex_array::builders::PrimitiveBuilder;
    use vortex_array::dtype::Nullability;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;

    use crate::BtrBlocksCompressor;

    #[test]
    fn test_constant_compressed() -> VortexResult<()> {
        let values: Vec<f64> = vec![42.5; 100];
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array.into_array())?;
        assert!(compressed.is::<Constant>());
        Ok(())
    }

    #[test]
    fn test_alp_compressed() -> VortexResult<()> {
        let values: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.01).collect();
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array.into_array())?;
        assert!(compressed.is::<ALP>());
        Ok(())
    }

    #[test]
    fn test_dict_compressed() -> VortexResult<()> {
        let distinct_values = [1.1, 2.2, 3.3, 4.4, 5.5];
        let values: Vec<f64> = (0..1000)
            .map(|i| distinct_values[i % distinct_values.len()])
            .collect();
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array.into_array())?;
        assert!(compressed.is::<ALP>());
        assert!(compressed.children()[0].is::<Dict>());
        Ok(())
    }

    #[test]
    fn test_null_dominated_compressed() -> VortexResult<()> {
        let mut builder = PrimitiveBuilder::<f64>::with_capacity(Nullability::Nullable, 100);
        for i in 0..5 {
            builder.append_value(i as f64);
        }
        builder.append_nulls(95);
        let array = builder.finish_into_primitive();
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array.into_array())?;
        // Verify the compressed array preserves values.
        assert_eq!(compressed.len(), 100);
        Ok(())
    }
}
