mod stats;

use vortex_alp::{alp_encode, ALPArray, RDEncoder};
use vortex_array::array::{ConstantArray, PrimitiveArray};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{Array, IntoArray, IntoArrayVariant};
use vortex_dict::{dict_encode, DictArray};
use vortex_dtype::PType;
use vortex_error::{vortex_panic, VortexExpect, VortexResult};
use vortex_runend::compress::runend_encode;
use vortex_runend::RunEndArray;

use self::stats::FloatStats;
use crate::integer::IntCompressor;
use crate::{estimate_compression_ratio_with_sampling, Compressor, CompressorStats, Scheme};

/// Threshold for the average run length in an array before we consider run-end encoding.
const RUN_END_THRESHOLD: u32 = 2;

pub trait FloatScheme: Scheme<StatsType = FloatStats> {}

impl<T> FloatScheme for T where T: Scheme<StatsType = FloatStats> {}

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
}

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

    fn code(&self) -> u8 {
        0
    }

    fn expected_compression_ratio(
        &self,
        _stats: &Self::StatsType,
        _is_sample: bool,
        _allowed_cascading: usize,
        _excludes: &[u8],
    ) -> VortexResult<f64> {
        Ok(1.0)
    }

    fn compress(
        &self,
        stats: &Self::StatsType,
        _is_sample: bool,
        _allowed_cascading: usize,
        _excludes: &[u8],
    ) -> VortexResult<Array> {
        Ok(stats.source().clone().into_array())
    }
}

impl Scheme for ConstantScheme {
    type StatsType = FloatStats;

    fn code(&self) -> u8 {
        1
    }

    fn expected_compression_ratio(
        &self,
        stats: &Self::StatsType,
        is_sample: bool,
        _allowed_cascading: usize,
        _excludes: &[u8],
    ) -> VortexResult<f64> {
        // Never select Constant when sampling
        if is_sample {
            return Ok(0.0);
        }

        // Can only have 1 distinct value
        if stats.distinct_value_count > 1 {
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
        _excludes: &[u8],
    ) -> VortexResult<Array> {
        let scalar = stats
            .source()
            .as_constant()
            .vortex_expect("must be constant");

        Ok(ConstantArray::new(scalar, stats.source().len()).into_array())
    }
}

impl Scheme for ALPScheme {
    type StatsType = FloatStats;

    fn code(&self) -> u8 {
        2
    }

    fn expected_compression_ratio(
        &self,
        stats: &Self::StatsType,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[u8],
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
        _excludes: &[u8],
    ) -> VortexResult<Array> {
        let alp = alp_encode(stats.source())?;
        let alp_ints = alp.encoded().into_primitive()?;

        // Compress the ALP ints.
        // Patches are not compressed. They should be infrequent, and if they are not then we want
        // to keep them linear for easy indexing.
        let compressed_alp_ints =
            IntCompressor::compress(&alp_ints, is_sample, allowed_cascading - 1, &[])?;

        Ok(ALPArray::try_new(compressed_alp_ints, alp.exponents(), alp.patches())?.into_array())
    }
}

impl Scheme for ALPRDScheme {
    type StatsType = FloatStats;

    fn code(&self) -> u8 {
        3
    }

    fn expected_compression_ratio(
        &self,
        stats: &Self::StatsType,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[u8],
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
        _excludes: &[u8],
    ) -> VortexResult<Array> {
        let encoder = match stats.source().ptype() {
            PType::F32 => RDEncoder::new(stats.source().as_slice::<f32>()),
            PType::F64 => RDEncoder::new(stats.source().as_slice::<f64>()),
            ptype => vortex_panic!("cannot ALPRD compress ptype {ptype}"),
        };

        Ok(encoder.encode(stats.source()).into_array())
    }
}

impl Scheme for DictScheme {
    type StatsType = FloatStats;

    fn code(&self) -> u8 {
        4
    }

    fn expected_compression_ratio(
        &self,
        stats: &Self::StatsType,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[u8],
    ) -> VortexResult<f64> {
        // Take a sample and run compression on the sample to determine before/after size.
        let sample = if is_sample {
            stats.clone()
        } else {
            stats.sample(64, 10)
        };

        let after = self
            .compress(&sample, is_sample, allowed_cascading, excludes)?
            .nbytes();
        let before = stats.source().nbytes();

        Ok(before as f64 / after as f64)
    }

    fn compress(
        &self,
        stats: &Self::StatsType,
        is_sample: bool,
        allowed_cascading: usize,
        _excludes: &[u8],
    ) -> VortexResult<Array> {
        let dict_array = dict_encode(stats.source())?;

        // Only compress the codes.
        let compressed_codes = IntCompressor::compress(
            &dict_array.codes().into_primitive()?,
            is_sample,
            allowed_cascading - 1,
            &[super::integer::DictScheme.code()],
        )?;

        Ok(DictArray::try_new(compressed_codes, dict_array.values())?.into_array())
    }
}

impl Scheme for RunEndScheme {
    type StatsType = FloatStats;

    fn code(&self) -> u8 {
        5
    }

    fn expected_compression_ratio(
        &self,
        stats: &Self::StatsType,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[u8],
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
        excludes: &[u8],
    ) -> VortexResult<Array> {
        let (ends, values) = runend_encode(stats.source())?;
        // Integer compress the ends, leave the values uncompressed.
        let compressed_ends =
            IntCompressor::compress(&ends, is_sample, allowed_cascading - 1, excludes)?;

        Ok(RunEndArray::try_new(compressed_ends, values)?.into_array())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::{IntoArray, IntoArrayVariant};
    use vortex_buffer::buffer_mut;

    use crate::float::FloatCompressor;
    use crate::{Compressor, MAX_CASCADE};

    #[test]
    fn test_compress() {
        let mut values = buffer_mut![1.0f32; 1024];
        // Sprinkle some other values in.
        for i in 0..1024 {
            // Insert 2.0 at all odd positions.
            // This should force dictionary encoding and exclude run-end due to the
            // average run length being 1.
            if i % 2 == 0 {
                values[i] = 2.0f32;
            }
        }

        let floats = values.into_array().into_primitive().unwrap();
        let compressed = FloatCompressor::compress(&floats, false, MAX_CASCADE, &[]).unwrap();
        println!("compressed: {}", compressed.tree_display())
    }
}
