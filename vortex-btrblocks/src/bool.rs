use itertools::Itertools;
use vortex_array::IntoArray;
use vortex_array::{
    ArrayRef, ToCanonical,
    arrays::{BoolArray, BoolVTable, PrimitiveArray},
    validity::Validity,
    vtable::ValidityHelper,
};
use vortex_compressedbool::CompressedBoolArray;
use vortex_dtype::PType;
use vortex_error::{VortexExpect as _, VortexResult};

use crate::{Compressor, CompressorStats, IntCompressor, Scheme, sample::sample};

/// [`Compressor`] for signed and unsigned integers.
pub struct BoolCompressor;

pub trait BoolScheme: Scheme<StatsType = BoolStats, CodeType = BoolCode> {}

// Auto-impl
impl<T> BoolScheme for T where T: Scheme<StatsType = BoolStats, CodeType = BoolCode> {}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct BoolCode(u8);

const UNCOMPRESSED_SCHEME: BoolCode = BoolCode(0);
const COMPRESSED_SCHEME: BoolCode = BoolCode(1);
const DICT_SCHEME: BoolCode = BoolCode(2);

impl Compressor for BoolCompressor {
    type ArrayVTable = BoolVTable;

    type SchemeType = dyn BoolScheme;

    type StatsType = BoolStats;

    fn schemes() -> &'static [&'static Self::SchemeType] {
        &[&UncompressedScheme as &dyn BoolScheme, &CompressedScheme]
    }

    fn default_scheme() -> &'static Self::SchemeType {
        &UncompressedScheme
    }

    fn dict_scheme_code() -> <Self::SchemeType as Scheme>::CodeType {
        // FIXME(DK): this seems wrong.
        DICT_SCHEME
    }
}

/// Array of integers and relevant stats for compression.
#[derive(Clone, Debug)]
pub struct BoolStats {
    pub(super) src: BoolArray,
    // cache for validity.false_count()
    pub(super) null_count: u32,
    // cache for validity.true_count()
    pub(super) true_count: u32,
    pub(super) false_count: u32,
}

impl CompressorStats for BoolStats {
    type ArrayVTable = BoolVTable;

    fn source(&self) -> &BoolArray {
        &self.src
    }

    fn generate_opts(
        input: &<Self::ArrayVTable as vortex_array::vtable::VTable>::Array,
        _opts: crate::GenerateStatsOptions,
    ) -> Self {
        match input.validity() {
            Validity::NonNullable => {
                let true_count = u32::try_from(input.bit_buffer().iter().filter(|x| *x).count())
                    .vortex_expect("array length fits in u32");
                let len = u32::try_from(input.len()).vortex_expect("array length fits in u32");
                BoolStats {
                    src: input.clone(),
                    null_count: 0,
                    true_count,
                    false_count: len - true_count,
                }
            }
            Validity::AllValid => {
                let true_count = u32::try_from(input.bit_buffer().iter().filter(|x| *x).count())
                    .vortex_expect("array length fits in u32");
                let len = u32::try_from(input.len()).vortex_expect("array length fits in u32");
                BoolStats {
                    src: input.clone(),
                    null_count: 0,
                    true_count,
                    false_count: len - true_count,
                }
            }
            Validity::AllInvalid => {
                let len = u32::try_from(input.len()).vortex_expect("array length fits in u32");
                BoolStats {
                    src: input.clone(),
                    null_count: len,
                    true_count: 0,
                    false_count: 0,
                }
            }
            Validity::Array(array) => {
                let (true_count, null_count) = input
                    .bit_buffer()
                    .iter()
                    .zip_eq(array.to_bool().bit_buffer().iter())
                    .map(|(is_true, is_valid)| ((is_true && is_valid) as u32, !is_valid as u32))
                    .fold((0, 0), |(l1, r1), (l2, r2)| (l1 + l2, r1 + r2));
                let len = u32::try_from(input.len()).vortex_expect("array length fits in u32");
                BoolStats {
                    src: input.clone(),
                    null_count,
                    true_count,
                    false_count: len - true_count,
                }
            }
        }
    }

    fn sample_opts(
        &self,
        sample_size: u32,
        sample_count: u32,
        opts: crate::GenerateStatsOptions,
    ) -> Self {
        let sampled = sample(self.src.as_ref(), sample_size, sample_count).to_bool();
        Self::generate_opts(&sampled, opts)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct UncompressedScheme;

impl Scheme for UncompressedScheme {
    type StatsType = BoolStats;
    type CodeType = BoolCode;

    fn code(&self) -> BoolCode {
        UNCOMPRESSED_SCHEME
    }

    fn expected_compression_ratio(
        &self,
        _stats: &BoolStats,
        _is_sample: bool,
        _allowed_cascading: usize,
        _excludes: &[BoolCode],
    ) -> VortexResult<f64> {
        // no compression
        Ok(1.0)
    }

    fn compress(
        &self,
        stats: &BoolStats,
        _is_sample: bool,
        _allowed_cascading: usize,
        _excludes: &[BoolCode],
    ) -> VortexResult<ArrayRef> {
        Ok(stats.source().to_array())
    }
}

#[derive(Debug, Copy, Clone)]
pub struct CompressedScheme;

impl Scheme for CompressedScheme {
    type StatsType = BoolStats;
    type CodeType = BoolCode;

    fn code(&self) -> BoolCode {
        COMPRESSED_SCHEME
    }

    fn compress(
        &self,
        stats: &Self::StatsType,
        _is_sample: bool,
        allowed_cascading: usize,
        _excludes: &[Self::CodeType],
    ) -> VortexResult<ArrayRef> {
        if allowed_cascading == 0 {
            return Ok(stats.source().to_array());
        }

        let array = stats.source();
        let validity = array.validity().clone();
        let bit_buffer = array.bit_buffer().clone();
        let offset = bit_buffer.offset();
        assert!(offset < 8);
        let len = bit_buffer.len();
        let byte_buffer = bit_buffer.into_inner();
        let fake_array =
            PrimitiveArray::from_byte_buffer(byte_buffer, PType::U8, Validity::NonNullable);
        let compressed = IntCompressor::compress(&fake_array, false, allowed_cascading - 1, &[])?;

        Ok(CompressedBoolArray::try_new(compressed, validity, offset, len)?.into_array())
    }
}
