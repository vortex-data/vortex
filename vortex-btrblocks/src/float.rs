// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(crate) mod dictionary;
mod stats;

use vortex_alp::{ALPArray, ALPEncoding, ALPVTable, RDEncoder};
use vortex_array::arrays::{ConstantArray, MaskedArray, PrimitiveVTable};
use vortex_array::vtable::ValidityHelper;
use vortex_array::{ArrayRef, IntoArray, ToCanonical};
use vortex_dict::DictArray;
use vortex_dtype::PType;
use vortex_error::{VortexExpect, VortexResult, vortex_panic};
use vortex_scalar::Scalar;
use vortex_sparse::{SparseArray, SparseVTable};

pub use self::stats::FloatStats;
use crate::float::dictionary::dictionary_encode;
use crate::integer::{IntCompressor, IntegerStats};
use crate::patches::compress_patches;
use crate::rle::RLEScheme;
use crate::{
    Compressor, CompressorStats, GenerateStatsOptions, Scheme,
    estimate_compression_ratio_with_sampling, integer,
};

pub trait FloatScheme: Scheme<StatsType = FloatStats, CodeType = FloatCode> {}

impl<T> FloatScheme for T where T: Scheme<StatsType = FloatStats, CodeType = FloatCode> {}

/// [`Compressor`] for floating-point numbers.
pub struct FloatCompressor;

impl Compressor for FloatCompressor {
    type ArrayVTable = PrimitiveVTable;
    type SchemeType = dyn FloatScheme;
    type StatsType = FloatStats;

    fn schemes() -> &'static [&'static Self::SchemeType] {
        &[
            &UncompressedScheme,
            &ConstantScheme,
            &ALPScheme,
            &ALPRDScheme,
            &DictScheme,
            &NullDominated,
            &RLE_FLOAT_SCHEME,
        ]
    }

    fn default_scheme() -> &'static Self::SchemeType {
        &UncompressedScheme
    }

    fn dict_scheme_code() -> FloatCode {
        DICT_SCHEME
    }
}

const UNCOMPRESSED_SCHEME: FloatCode = FloatCode(0);
const CONSTANT_SCHEME: FloatCode = FloatCode(1);
const ALP_SCHEME: FloatCode = FloatCode(2);
const ALPRD_SCHEME: FloatCode = FloatCode(3);
const DICT_SCHEME: FloatCode = FloatCode(4);
const RUN_END_SCHEME: FloatCode = FloatCode(5);
const RUN_LENGTH_SCHEME: FloatCode = FloatCode(6);

const SPARSE_SCHEME: FloatCode = FloatCode(7);

#[derive(Debug, Copy, Clone)]
struct UncompressedScheme;

#[derive(Debug, Copy, Clone)]
struct ConstantScheme;

#[derive(Debug, Copy, Clone)]
struct ALPScheme;

#[derive(Debug, Copy, Clone)]
struct ALPRDScheme;

#[derive(Debug, Copy, Clone)]
struct DictScheme;

#[derive(Debug, Copy, Clone)]
pub struct NullDominated;

pub const RLE_FLOAT_SCHEME: RLEScheme<FloatStats, FloatCode> = RLEScheme::new(
    RUN_LENGTH_SCHEME,
    |values, is_sample, allowed_cascading, excludes| {
        FloatCompressor::compress(values, is_sample, allowed_cascading, excludes)
    },
);

impl Scheme for UncompressedScheme {
    type StatsType = FloatStats;
    type CodeType = FloatCode;

    fn code(&self) -> FloatCode {
        UNCOMPRESSED_SCHEME
    }

    fn expected_compression_ratio(
        &self,
        _stats: &Self::StatsType,
        _is_sample: bool,
        _allowed_cascading: usize,
        _excludes: &[FloatCode],
    ) -> VortexResult<f64> {
        Ok(1.0)
    }

    fn compress(
        &self,
        stats: &Self::StatsType,
        _is_sample: bool,
        _allowed_cascading: usize,
        _excludes: &[FloatCode],
    ) -> VortexResult<ArrayRef> {
        Ok(stats.source().to_array())
    }
}

impl Scheme for ConstantScheme {
    type StatsType = FloatStats;
    type CodeType = FloatCode;

    fn code(&self) -> FloatCode {
        CONSTANT_SCHEME
    }

    fn expected_compression_ratio(
        &self,
        stats: &Self::StatsType,
        is_sample: bool,
        _allowed_cascading: usize,
        _excludes: &[FloatCode],
    ) -> VortexResult<f64> {
        // Never select Constant when sampling
        if is_sample {
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
        stats: &Self::StatsType,
        _is_sample: bool,
        _allowed_cascading: usize,
        _excludes: &[FloatCode],
    ) -> VortexResult<ArrayRef> {
        let scalar_idx = (0..stats.source().len()).position(|idx| stats.source().is_valid(idx));

        match scalar_idx {
            Some(idx) => {
                let scalar = stats.source().scalar_at(idx);
                let const_arr = ConstantArray::new(scalar, stats.src.len()).into_array();
                if !stats.source().all_valid() {
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

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct FloatCode(u8);

impl Scheme for ALPScheme {
    type StatsType = FloatStats;
    type CodeType = FloatCode;

    fn code(&self) -> FloatCode {
        ALP_SCHEME
    }

    fn expected_compression_ratio(
        &self,
        stats: &Self::StatsType,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[FloatCode],
    ) -> VortexResult<f64> {
        // We don't support ALP for f16
        if stats.source().ptype() == PType::F16 {
            return Ok(0.0);
        }

        if allowed_cascading == 0 {
            // ALP does not compress on its own, we need to be able to cascade it with
            // an integer compressor.
            return Ok(0.0);
        }

        estimate_compression_ratio_with_sampling(
            self,
            stats,
            is_sample,
            allowed_cascading,
            excludes,
        )
    }

    fn compress(
        &self,
        stats: &FloatStats,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[FloatCode],
    ) -> VortexResult<ArrayRef> {
        let alp_encoded = ALPEncoding
            .encode(&stats.source().to_canonical(), None)?
            .vortex_expect("Input is a supported floating point array");
        let alp = alp_encoded.as_::<ALPVTable>();
        let alp_ints = alp.encoded().to_primitive();

        // Compress the ALP ints.
        // Patches are not compressed. They should be infrequent, and if they are not then we want
        // to keep them linear for easy indexing.
        let mut int_excludes = Vec::new();
        if excludes.contains(&DICT_SCHEME) {
            int_excludes.push(integer::DictScheme.code());
        }
        if excludes.contains(&RUN_END_SCHEME) {
            int_excludes.push(integer::RunEndScheme.code());
        }

        let compressed_alp_ints =
            IntCompressor::compress(&alp_ints, is_sample, allowed_cascading - 1, &int_excludes)?;

        let patches = alp.patches().map(compress_patches).transpose()?;

        Ok(ALPArray::new(compressed_alp_ints, alp.exponents(), patches).into_array())
    }
}

impl Scheme for ALPRDScheme {
    type StatsType = FloatStats;
    type CodeType = FloatCode;

    fn code(&self) -> FloatCode {
        ALPRD_SCHEME
    }

    fn expected_compression_ratio(
        &self,
        stats: &Self::StatsType,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[FloatCode],
    ) -> VortexResult<f64> {
        if stats.source().ptype() == PType::F16 {
            return Ok(0.0);
        }

        estimate_compression_ratio_with_sampling(
            self,
            stats,
            is_sample,
            allowed_cascading,
            excludes,
        )
    }

    fn compress(
        &self,
        stats: &Self::StatsType,
        _is_sample: bool,
        _allowed_cascading: usize,
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
        DICT_SCHEME
    }

    fn expected_compression_ratio(
        &self,
        stats: &Self::StatsType,
        is_sample: bool,
        allowed_cascading: usize,
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
        estimate_compression_ratio_with_sampling(
            self,
            stats,
            is_sample,
            allowed_cascading,
            excludes,
        )
    }

    fn compress(
        &self,
        stats: &Self::StatsType,
        is_sample: bool,
        allowed_cascading: usize,
        _excludes: &[FloatCode],
    ) -> VortexResult<ArrayRef> {
        let dict_array = dictionary_encode(stats);

        // Only compress the codes.
        let codes_stats = IntegerStats::generate_opts(
            &dict_array.codes().to_primitive().narrow()?,
            GenerateStatsOptions {
                count_distinct_values: false,
            },
        );
        let codes_scheme = IntCompressor::choose_scheme(
            &codes_stats,
            is_sample,
            allowed_cascading - 1,
            &[integer::DictScheme.code(), integer::SequenceScheme.code()],
        )?;
        let compressed_codes = codes_scheme.compress(
            &codes_stats,
            is_sample,
            allowed_cascading - 1,
            &[integer::DictScheme.code()],
        )?;

        let compressed_values = FloatCompressor::compress(
            &dict_array.values().to_primitive(),
            is_sample,
            allowed_cascading - 1,
            &[DICT_SCHEME],
        )?;

        // SAFETY: compressing codes or values does not alter the invariants
        unsafe { Ok(DictArray::new_unchecked(compressed_codes, compressed_values).into_array()) }
    }
}

impl Scheme for NullDominated {
    type StatsType = FloatStats;
    type CodeType = FloatCode;

    fn code(&self) -> Self::CodeType {
        SPARSE_SCHEME
    }

    fn expected_compression_ratio(
        &self,
        stats: &Self::StatsType,
        _is_sample: bool,
        allowed_cascading: usize,
        _excludes: &[Self::CodeType],
    ) -> VortexResult<f64> {
        // Only use `SparseScheme` if we can cascade.
        if allowed_cascading == 0 {
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
        stats: &Self::StatsType,
        is_sample: bool,
        allowed_cascading: usize,
        _excludes: &[Self::CodeType],
    ) -> VortexResult<ArrayRef> {
        assert!(allowed_cascading > 0);

        // We pass None as we only run this pathway for NULL-dominated float arrays
        let sparse_encoded = SparseArray::encode(stats.src.as_ref(), None)?;

        if let Some(sparse) = sparse_encoded.as_opt::<SparseVTable>() {
            // Compress the values
            let new_excludes = vec![integer::SparseScheme.code()];

            // Don't attempt to compress the non-null values

            let indices = sparse.patches().indices().to_primitive().narrow()?;
            let compressed_indices = IntCompressor::compress_no_dict(
                &indices,
                is_sample,
                allowed_cascading - 1,
                &new_excludes,
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

#[cfg(test)]
mod tests {
    use std::iter;

    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::builders::{ArrayBuilder, PrimitiveBuilder};
    use vortex_array::validity::Validity;
    use vortex_array::{Array, IntoArray, ToCanonical, assert_arrays_eq};
    use vortex_buffer::{Buffer, buffer_mut};
    use vortex_dtype::Nullability;
    use vortex_sparse::SparseEncoding;

    use crate::float::{FloatCompressor, RLE_FLOAT_SCHEME};
    use crate::{Compressor, CompressorStats, MAX_CASCADE, Scheme};

    #[test]
    fn test_empty() {
        // Make sure empty array compression does not fail
        let result = FloatCompressor::compress(
            &PrimitiveArray::new(Buffer::<f32>::empty(), Validity::NonNullable),
            false,
            3,
            &[],
        )
        .unwrap();

        assert!(result.is_empty());
    }

    #[test]
    fn test_compress() {
        let mut values = buffer_mut![1.0f32; 1024];
        // Sprinkle some other values in.
        for i in 0..1024 {
            // Insert 2.0 at all odd positions.
            // This should force dictionary encoding and exclude run-end due to the
            // average run length being 1.
            values[i] = (i % 50) as f32;
        }

        let floats = values.into_array().to_primitive();
        let compressed = FloatCompressor::compress(&floats, false, MAX_CASCADE, &[]).unwrap();
        println!("compressed: {}", compressed.display_tree())
    }

    #[test]
    fn test_rle_compression() {
        let mut values = Vec::new();
        values.extend(iter::repeat_n(1.5f32, 100));
        values.extend(iter::repeat_n(2.7f32, 200));
        values.extend(iter::repeat_n(3.15f32, 150));

        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let stats = crate::float::FloatStats::generate(&array);
        let compressed = RLE_FLOAT_SCHEME.compress(&stats, false, 3, &[]).unwrap();

        let decoded = compressed;
        let expected = Buffer::copy_from(&values).into_array();
        assert_arrays_eq!(decoded.as_ref(), expected.as_ref());
    }

    #[test]
    fn test_sparse_compression() {
        let mut array = PrimitiveBuilder::<f32>::with_capacity(Nullability::Nullable, 100);
        array.append_value(f32::NAN);
        array.append_value(-f32::NAN);
        array.append_value(f32::INFINITY);
        array.append_value(-f32::INFINITY);
        array.append_value(0.0f32);
        array.append_value(-0.0f32);
        array.append_nulls(90);

        let floats = array.finish_into_primitive();

        let compressed = FloatCompressor::compress(&floats, false, MAX_CASCADE, &[]).unwrap();

        assert_eq!(compressed.encoding_id(), SparseEncoding.id());
    }
}
