// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(crate) mod dictionary;
pub(super) mod stats;

use std::hash::Hash;
use std::hash::Hasher;

use enum_iterator::Sequence;
use vortex_alp::ALPArray;
use vortex_alp::ALPVTable;
use vortex_alp::RDEncoder;
use vortex_alp::alp_encode;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::DictArrayParts;
use vortex_array::arrays::MaskedArray;
use vortex_array::arrays::PrimitiveVTable;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityHelper;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_sparse::SparseArray;
use vortex_sparse::SparseVTable;

use self::dictionary::dictionary_encode;
pub use self::stats::FloatStats;
use super::integer::DictScheme as IntDictScheme;
use super::integer::RunEndScheme as IntRunEndScheme;
use super::integer::SparseScheme as IntSparseScheme;
use crate::BtrBlocksCompressor;
use crate::CanonicalCompressor;
use crate::Compressor;
use crate::CompressorContext;
use crate::CompressorStats;
use crate::Excludes;
use crate::GenerateStatsOptions;
use crate::IntCode;
use crate::Scheme;
use crate::SchemeExt;
use crate::compressor::patches::compress_patches;
use crate::compressor::rle;
use crate::compressor::rle::RLEScheme;

pub trait FloatScheme: Scheme<StatsType = FloatStats, CodeType = FloatCode> + Send + Sync {}

impl<T> FloatScheme for T where T: Scheme<StatsType = FloatStats, CodeType = FloatCode> + Send + Sync
{}

impl PartialEq for dyn FloatScheme {
    fn eq(&self, other: &Self) -> bool {
        self.code() == other.code()
    }
}

impl Eq for dyn FloatScheme {}

impl Hash for dyn FloatScheme {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.code().hash(state)
    }
}

/// All available float compression schemes.
pub const ALL_FLOAT_SCHEMES: &[&dyn FloatScheme] = &[
    &UncompressedScheme,
    &ConstantScheme,
    &ALPScheme,
    &ALPRDScheme,
    &DictScheme,
    &NullDominated,
    &RLE_FLOAT_SCHEME,
    #[cfg(feature = "pco")]
    &PcoScheme,
];

/// [`Compressor`] for floating-point numbers.
#[derive(Clone, Copy)]
pub struct FloatCompressor<'a> {
    /// Reference to the parent compressor.
    pub btr_blocks_compressor: &'a dyn CanonicalCompressor,
}

impl<'a> Compressor for FloatCompressor<'a> {
    type ArrayVTable = PrimitiveVTable;
    type SchemeType = dyn FloatScheme;
    type StatsType = FloatStats;

    fn gen_stats(&self, array: &<Self::ArrayVTable as VTable>::Array) -> Self::StatsType {
        if self
            .btr_blocks_compressor
            .float_schemes()
            .iter()
            .any(|s| s.code() == DictScheme.code())
        {
            FloatStats::generate_opts(
                array,
                GenerateStatsOptions {
                    count_distinct_values: true,
                },
            )
        } else {
            FloatStats::generate_opts(
                array,
                GenerateStatsOptions {
                    count_distinct_values: false,
                },
            )
        }
    }

    fn schemes(&self) -> &[&'static dyn FloatScheme] {
        self.btr_blocks_compressor.float_schemes()
    }

    fn default_scheme(&self) -> &'static Self::SchemeType {
        &UncompressedScheme
    }
}

/// Unique identifier for float compression schemes.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Sequence, Ord, PartialOrd)]
pub enum FloatCode {
    /// No compression applied.
    Uncompressed,
    /// Constant encoding for arrays with a single distinct value.
    Constant,
    /// ALP (Adaptive Lossless floating-Point) encoding.
    Alp,
    /// ALPRD (ALP with Right Division) encoding variant.
    AlpRd,
    /// Dictionary encoding for low-cardinality float values.
    Dict,
    /// Run-end encoding.
    RunEnd,
    /// RLE encoding - generic run-length encoding.
    Rle,
    /// Sparse encoding for null-dominated arrays.
    Sparse,
    /// Pco (pcodec) compression for floats.
    Pco,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct UncompressedScheme;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct ConstantScheme;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct ALPScheme;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct ALPRDScheme;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct DictScheme;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NullDominated;

/// Pco (pcodec) compression for floats.
#[cfg(feature = "pco")]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct PcoScheme;

/// Configuration for float RLE compression.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FloatRLEConfig;

impl rle::RLEConfig for FloatRLEConfig {
    type Stats = FloatStats;
    type Code = FloatCode;

    const CODE: FloatCode = FloatCode::Rle;

    fn compress_values(
        compressor: &BtrBlocksCompressor,
        values: &vortex_array::arrays::PrimitiveArray,
        ctx: CompressorContext,
        excludes: &[FloatCode],
    ) -> VortexResult<ArrayRef> {
        compressor.compress_canonical(Canonical::Primitive(values.clone()), ctx, excludes.into())
    }
}

/// RLE scheme for float compression.
pub const RLE_FLOAT_SCHEME: RLEScheme<FloatRLEConfig> = RLEScheme::new();

impl Scheme for UncompressedScheme {
    type StatsType = FloatStats;
    type CodeType = FloatCode;

    fn code(&self) -> FloatCode {
        FloatCode::Uncompressed
    }

    fn expected_compression_ratio(
        &self,
        _compressor: &BtrBlocksCompressor,
        _stats: &Self::StatsType,
        _ctx: CompressorContext,
        _excludes: &[FloatCode],
    ) -> VortexResult<f64> {
        Ok(1.0)
    }

    fn compress(
        &self,
        _btr_blocks_compressor: &BtrBlocksCompressor,
        stats: &Self::StatsType,
        _ctx: CompressorContext,
        _excludes: &[FloatCode],
    ) -> VortexResult<ArrayRef> {
        Ok(stats.source().to_array())
    }
}

impl Scheme for ConstantScheme {
    type StatsType = FloatStats;
    type CodeType = FloatCode;

    fn code(&self) -> FloatCode {
        FloatCode::Constant
    }

    fn expected_compression_ratio(
        &self,
        _btr_blocks_compressor: &BtrBlocksCompressor,
        stats: &Self::StatsType,
        ctx: CompressorContext,
        _excludes: &[FloatCode],
    ) -> VortexResult<f64> {
        // Never select Constant when sampling
        if ctx.is_sample {
            return Ok(0.0);
        }

        if stats.null_count as usize == stats.src.len() || stats.value_count == 0 {
            return Ok(0.0);
        }

        // Can only have 1 distinct value
        if stats.distinct_values_count != 1 {
            return Ok(0.0);
        }

        Ok(stats.value_count as f64)
    }

    fn compress(
        &self,
        _btr_blocks_compressor: &BtrBlocksCompressor,
        stats: &Self::StatsType,
        _ctx: CompressorContext,
        _excludes: &[FloatCode],
    ) -> VortexResult<ArrayRef> {
        let scalar_idx =
            (0..stats.source().len()).position(|idx| stats.source().is_valid(idx).unwrap_or(false));

        match scalar_idx {
            Some(idx) => {
                let scalar = stats.source().scalar_at(idx)?;
                let const_arr = ConstantArray::new(scalar, stats.src.len()).into_array();
                if !stats.source().all_valid()? {
                    Ok(MaskedArray::try_new(const_arr, stats.src.validity().clone())?.into_array())
                } else {
                    Ok(const_arr)
                }
            }
            None => Ok(ConstantArray::new(
                Scalar::null(stats.src.dtype().clone()),
                stats.src.len(),
            )
            .into_array()),
        }
    }
}

impl Scheme for ALPScheme {
    type StatsType = FloatStats;
    type CodeType = FloatCode;

    fn code(&self) -> FloatCode {
        FloatCode::Alp
    }

    fn expected_compression_ratio(
        &self,
        compressor: &BtrBlocksCompressor,
        stats: &Self::StatsType,
        ctx: CompressorContext,
        excludes: &[FloatCode],
    ) -> VortexResult<f64> {
        // We don't support ALP for f16
        if stats.source().ptype() == PType::F16 {
            return Ok(0.0);
        }

        if ctx.allowed_cascading == 0 {
            // ALP does not compress on its own, we need to be able to cascade it with
            // an integer compressor.
            return Ok(0.0);
        }

        self.estimate_compression_ratio_with_sampling(compressor, stats, ctx, excludes)
    }

    fn compress(
        &self,
        compressor: &BtrBlocksCompressor,
        stats: &FloatStats,
        ctx: CompressorContext,
        excludes: &[FloatCode],
    ) -> VortexResult<ArrayRef> {
        let alp_encoded = alp_encode(&stats.source().to_primitive(), None)?;
        let alp = alp_encoded.as_::<ALPVTable>();
        let alp_ints = alp.encoded().to_primitive();

        // Compress the ALP ints.
        // Patches are not compressed. They should be infrequent, and if they are not then we want
        // to keep them linear for easy indexing.
        let mut int_excludes = Vec::new();
        if excludes.contains(&FloatCode::Dict) {
            int_excludes.push(IntDictScheme.code());
        }
        if excludes.contains(&FloatCode::RunEnd) {
            int_excludes.push(IntRunEndScheme.code());
        }

        let compressed_alp_ints = compressor.compress_canonical(
            Canonical::Primitive(alp_ints),
            ctx.descend(),
            Excludes::int_only(&int_excludes),
        )?;

        let patches = alp.patches().map(compress_patches).transpose()?;

        Ok(ALPArray::new(compressed_alp_ints, alp.exponents(), patches).into_array())
    }
}

impl Scheme for ALPRDScheme {
    type StatsType = FloatStats;
    type CodeType = FloatCode;

    fn code(&self) -> FloatCode {
        FloatCode::AlpRd
    }

    fn expected_compression_ratio(
        &self,
        compressor: &BtrBlocksCompressor,
        stats: &Self::StatsType,
        ctx: CompressorContext,
        excludes: &[FloatCode],
    ) -> VortexResult<f64> {
        if stats.source().ptype() == PType::F16 {
            return Ok(0.0);
        }

        self.estimate_compression_ratio_with_sampling(compressor, stats, ctx, excludes)
    }

    fn compress(
        &self,
        _compressor: &BtrBlocksCompressor,
        stats: &Self::StatsType,
        _ctx: CompressorContext,
        _excludes: &[FloatCode],
    ) -> VortexResult<ArrayRef> {
        let encoder = match stats.source().ptype() {
            PType::F32 => RDEncoder::new(stats.source().as_slice::<f32>()),
            PType::F64 => RDEncoder::new(stats.source().as_slice::<f64>()),
            ptype => vortex_panic!("cannot ALPRD compress ptype {ptype}"),
        };

        let mut alp_rd = encoder.encode(stats.source());

        let patches = alp_rd
            .left_parts_patches()
            .map(compress_patches)
            .transpose()?;
        alp_rd.replace_left_parts_patches(patches);

        Ok(alp_rd.into_array())
    }
}

impl Scheme for DictScheme {
    type StatsType = FloatStats;
    type CodeType = FloatCode;

    fn code(&self) -> FloatCode {
        FloatCode::Dict
    }

    fn expected_compression_ratio(
        &self,
        compressor: &BtrBlocksCompressor,
        stats: &Self::StatsType,
        ctx: CompressorContext,
        excludes: &[FloatCode],
    ) -> VortexResult<f64> {
        if stats.value_count == 0 {
            return Ok(0.0);
        }

        // If the array is high cardinality (>50% unique values) skip.
        if stats.distinct_values_count > stats.value_count / 2 {
            return Ok(0.0);
        }

        // Take a sample and run compression on the sample to determine before/after size.
        self.estimate_compression_ratio_with_sampling(compressor, stats, ctx, excludes)
    }

    fn compress(
        &self,
        compressor: &BtrBlocksCompressor,
        stats: &Self::StatsType,
        ctx: CompressorContext,
        _excludes: &[Self::CodeType],
    ) -> VortexResult<ArrayRef> {
        let dict = dictionary_encode(stats);
        let has_all_values_referenced = dict.has_all_values_referenced();
        let DictArrayParts { codes, values, .. } = dict.into_parts();

        let compressed_codes = compressor.compress_canonical(
            Canonical::Primitive(codes.to_primitive()),
            ctx.descend(),
            Excludes::int_only(&[IntCode::Dict, IntCode::Sequence]),
        )?;

        assert!(values.is_canonical());
        let compressed_values = compressor.compress_canonical(
            Canonical::Primitive(values.to_primitive()),
            ctx.descend(),
            Excludes::from(&[FloatCode::Dict]),
        )?;

        // SAFETY: compressing codes or values does not alter the invariants
        unsafe {
            Ok(
                DictArray::new_unchecked(compressed_codes, compressed_values)
                    .set_all_values_referenced(has_all_values_referenced)
                    .into_array(),
            )
        }
    }
}

impl Scheme for NullDominated {
    type StatsType = FloatStats;
    type CodeType = FloatCode;

    fn code(&self) -> Self::CodeType {
        FloatCode::Sparse
    }

    fn expected_compression_ratio(
        &self,
        _compressor: &BtrBlocksCompressor,
        stats: &Self::StatsType,
        ctx: CompressorContext,
        _excludes: &[Self::CodeType],
    ) -> VortexResult<f64> {
        // Only use `SparseScheme` if we can cascade.
        if ctx.allowed_cascading == 0 {
            return Ok(0.0);
        }

        if stats.value_count == 0 {
            // All nulls should use ConstantScheme
            return Ok(0.0);
        }

        // If the majority is null, will compress well.
        if stats.null_count as f64 / stats.src.len() as f64 > 0.9 {
            return Ok(stats.src.len() as f64 / stats.value_count as f64);
        }

        // Otherwise we don't go this route
        Ok(0.0)
    }

    fn compress(
        &self,
        compressor: &BtrBlocksCompressor,
        stats: &Self::StatsType,
        ctx: CompressorContext,
        _excludes: &[Self::CodeType],
    ) -> VortexResult<ArrayRef> {
        assert!(ctx.allowed_cascading > 0);

        // We pass None as we only run this pathway for NULL-dominated float arrays
        let sparse_encoded = SparseArray::encode(&stats.src.to_array(), None)?;

        if let Some(sparse) = sparse_encoded.as_opt::<SparseVTable>() {
            // Compress the values
            let new_excludes = [IntSparseScheme.code()];

            // Don't attempt to compress the non-null values

            let indices = sparse.patches().indices().to_primitive().narrow()?;
            let compressed_indices = compressor.compress_canonical(
                Canonical::Primitive(indices.to_primitive()),
                ctx.descend(),
                Excludes::int_only(&new_excludes),
            )?;

            SparseArray::try_new(
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
    type StatsType = FloatStats;
    type CodeType = FloatCode;

    fn code(&self) -> FloatCode {
        FloatCode::Pco
    }

    fn compress(
        &self,
        _compressor: &BtrBlocksCompressor,
        stats: &Self::StatsType,
        _ctx: CompressorContext,
        _excludes: &[FloatCode],
    ) -> VortexResult<ArrayRef> {
        Ok(vortex_pco::PcoArray::from_primitive(
            stats.source(),
            pco::DEFAULT_COMPRESSION_LEVEL,
            8192,
        )?
        .into_array())
    }
}

#[cfg(test)]
mod tests {

    use std::iter;

    use vortex_array::DynArray;
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builders::ArrayBuilder;
    use vortex_array::builders::PrimitiveBuilder;
    use vortex_array::display::DisplayOptions;
    use vortex_array::dtype::Nullability;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer_mut;
    use vortex_error::VortexResult;

    use super::RLE_FLOAT_SCHEME;
    use crate::BtrBlocksCompressor;
    use crate::CompressorContext;
    use crate::CompressorExt;
    use crate::CompressorStats;
    use crate::Scheme;

    #[test]
    fn test_empty() -> VortexResult<()> {
        // Make sure empty array compression does not fail
        let btr = BtrBlocksCompressor::default();
        let result = btr.float_compressor().compress(
            &btr,
            &PrimitiveArray::new(Buffer::<f32>::empty(), Validity::NonNullable),
            CompressorContext::default(),
            &[],
        )?;

        assert!(result.is_empty());
        Ok(())
    }

    #[test]
    fn test_compress() -> VortexResult<()> {
        let mut values = buffer_mut![1.0f32; 1024];
        // Sprinkle some other values in.
        for i in 0..1024 {
            // Insert 2.0 at all odd positions.
            // This should force dictionary encoding and exclude run-end due to the
            // average run length being 1.
            values[i] = (i % 50) as f32;
        }

        let floats = values.into_array().to_primitive();
        let btr = BtrBlocksCompressor::default();
        let compressed =
            btr.float_compressor()
                .compress(&btr, &floats, CompressorContext::default(), &[])?;
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
        let stats = super::FloatStats::generate(&array);
        let btr = BtrBlocksCompressor::default();
        let compressed =
            RLE_FLOAT_SCHEME.compress(&btr, &stats, CompressorContext::default(), &[])?;

        let decoded = compressed;
        let expected = Buffer::copy_from(&values).into_array();
        assert_arrays_eq!(decoded.as_ref(), expected.as_ref());
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

        let floats = array.finish_into_primitive();
        let btr = BtrBlocksCompressor::default();
        let compressed =
            btr.float_compressor()
                .compress(&btr, &floats, CompressorContext::default(), &[])?;
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

    use vortex_alp::ALPVTable;
    use vortex_array::arrays::ConstantVTable;
    use vortex_array::arrays::DictVTable;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::builders::ArrayBuilder;
    use vortex_array::builders::PrimitiveBuilder;
    use vortex_array::dtype::Nullability;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;

    use crate::BtrBlocksCompressor;
    use crate::CompressorContext;
    use crate::CompressorExt;

    #[test]
    fn test_constant_compressed() -> VortexResult<()> {
        let values: Vec<f64> = vec![42.5; 100];
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed =
            btr.float_compressor()
                .compress(&btr, &array, CompressorContext::default(), &[])?;
        assert!(compressed.is::<ConstantVTable>());
        Ok(())
    }

    #[test]
    fn test_alp_compressed() -> VortexResult<()> {
        let values: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.01).collect();
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed =
            btr.float_compressor()
                .compress(&btr, &array, CompressorContext::default(), &[])?;
        assert!(compressed.is::<ALPVTable>());
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
        let compressed =
            btr.float_compressor()
                .compress(&btr, &array, CompressorContext::default(), &[])?;
        assert!(compressed.is::<DictVTable>());
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
        let compressed =
            btr.float_compressor()
                .compress(&btr, &array, CompressorContext::default(), &[])?;
        // Verify the compressed array preserves values.
        assert_eq!(compressed.len(), 100);
        Ok(())
    }
}
