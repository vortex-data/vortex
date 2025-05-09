mod dictionary;
mod stats;

use vortex_alp::{ALPArray, ALPEncoding, RDEncoder};
use vortex_array::arrays::{ConstantArray, PrimitiveArray};
use vortex_array::vtable::EncodingVTable;
use vortex_array::{Array, ArrayExt as _, ArrayRef, ArrayStatistics, ToCanonical};
use vortex_dict::DictArray;
use vortex_dtype::PType;
use vortex_error::{VortexExpect, VortexResult, vortex_panic};
use vortex_runend::RunEndArray;
use vortex_runend::compress::runend_encode;

use self::stats::FloatStats;
use crate::float::dictionary::dictionary_encode;
use crate::integer::{IntCompressor, IntegerStats};
use crate::patches::compress_patches;
use crate::{
    Compressor, CompressorStats, GenerateStatsOptions, Scheme,
    estimate_compression_ratio_with_sampling, integer,
};

/// Threshold for the average run length in an array before we consider run-end encoding.
const RUN_END_THRESHOLD: u32 = 3;

pub trait FloatScheme: Scheme<StatsType = FloatStats, CodeType = FloatCode> {}

impl<T> FloatScheme for T where T: Scheme<StatsType = FloatStats, CodeType = FloatCode> {}

pub struct FloatCompressor;

impl Compressor for FloatCompressor {
    type ArrayType = PrimitiveArray;
    type SchemeType = dyn FloatScheme;
    type StatsType = FloatStats;

    fn schemes() -> &'static [&'static Self::SchemeType] {
        &[
            &UncompressedScheme,
            &ConstantScheme,
            &ALPScheme,
            &ALPRDScheme,
            &DictScheme,
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
const RUNEND_SCHEME: FloatCode = FloatCode(5);

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
struct RunEndScheme;

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

        // Can only have 1 distinct value
        if stats.distinct_values_count > 1 {
            return Ok(0.0);
        }

        // Cannot have mix of nulls and non-nulls
        if stats.null_count > 0 && stats.value_count > 0 {
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
        let scalar = stats
            .source()
            .as_constant()
            .vortex_expect("must be constant");

        Ok(ConstantArray::new(scalar, stats.source().len()).into_array())
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
            .encode(&stats.source().to_canonical()?, None)?
            .vortex_expect("Input is a supported floating point array");
        let alp = alp_encoded.as_::<ALPArray>();
        let alp_ints = alp.encoded().to_primitive()?;

        // Compress the ALP ints.
        // Patches are not compressed. They should be infrequent, and if they are not then we want
        // to keep them linear for easy indexing.
        let mut int_excludes = Vec::new();
        if excludes.contains(&DICT_SCHEME) {
            int_excludes.push(integer::DictScheme.code());
        }
        if excludes.contains(&RUNEND_SCHEME) {
            int_excludes.push(integer::RunEndScheme.code());
        }

        let compressed_alp_ints =
            IntCompressor::compress(&alp_ints, is_sample, allowed_cascading - 1, &int_excludes)?;

        let patches = alp.patches().map(compress_patches).transpose()?;

        Ok(ALPArray::try_new(compressed_alp_ints, alp.exponents(), patches)?.into_array())
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
        let dict_array = dictionary_encode(stats)?;

        // Only compress the codes.
        let codes_stats = IntegerStats::generate_opts(
            &dict_array.codes().to_primitive()?,
            GenerateStatsOptions {
                count_distinct_values: false,
            },
        );
        let codes_scheme = IntCompressor::choose_scheme(
            &codes_stats,
            is_sample,
            allowed_cascading - 1,
            &[integer::DictScheme.code()],
        )?;
        let compressed_codes = codes_scheme.compress(
            &codes_stats,
            is_sample,
            allowed_cascading - 1,
            &[integer::DictScheme.code()],
        )?;

        let compressed_values = FloatCompressor::compress(
            &dict_array.values().to_primitive()?,
            is_sample,
            allowed_cascading - 1,
            &[DICT_SCHEME],
        )?;

        Ok(DictArray::try_new(compressed_codes, compressed_values)?.into_array())
    }
}

impl Scheme for RunEndScheme {
    type StatsType = FloatStats;
    type CodeType = FloatCode;

    fn code(&self) -> FloatCode {
        RUNEND_SCHEME
    }

    fn expected_compression_ratio(
        &self,
        stats: &Self::StatsType,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[FloatCode],
    ) -> VortexResult<f64> {
        if stats.average_run_length < RUN_END_THRESHOLD {
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
        _excludes: &[FloatCode],
    ) -> VortexResult<ArrayRef> {
        let (ends, values) = runend_encode(stats.source())?;
        // Integer compress the ends, leave the values uncompressed.
        let compressed_ends = IntCompressor::compress(
            &ends,
            is_sample,
            allowed_cascading - 1,
            &[
                integer::RunEndScheme.code(),
                integer::DictScheme.code(),
                integer::SparseScheme.code(),
            ],
        )?;

        Ok(RunEndArray::try_new(compressed_ends, values)?.into_array())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_array::{Array, IntoArray, ToCanonical};
    use vortex_buffer::{Buffer, buffer_mut};

    use crate::float::FloatCompressor;
    use crate::{Compressor, MAX_CASCADE};

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

        let floats = values.into_array().to_primitive().unwrap();
        let compressed = FloatCompressor::compress(&floats, false, MAX_CASCADE, &[]).unwrap();
        println!("compressed: {}", compressed.tree_display())
    }
}
