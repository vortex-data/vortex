use vortex_array::aliases::hash_set::HashSet;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::{Array, IntoArray, IntoArrayVariant};
use vortex_dict::builders::dict_encode;
use vortex_dict::DictArray;
use vortex_error::{VortexExpect, VortexResult};
use vortex_fsst::{fsst_compress, fsst_train_compressor};

use crate::downscale::downscale_integer_array;
use crate::integer::IntCompressor;
use crate::sample::sample;
use crate::{
    estimate_compression_ratio_with_sampling, Compressor, CompressorStats, GenerateStatsOptions,
    Scheme,
};

#[derive(Clone)]
pub struct StringStats {
    src: VarBinViewArray,
    estimated_distinct_count: u32,
    value_count: u32,
    // null_count: u32,
}

/// Estimate the number of distinct strings in the var bin view array.
#[allow(clippy::cast_possible_truncation)]
fn estimate_distinct_count(strings: &VarBinViewArray) -> u32 {
    let views = strings.views();
    // Iterate the views. Two strings which are equal must have the same first 8-bytes.
    // NOTE: there are cases where this performs pessimally, e.g. when we have strings that all
    // share a 4-byte prefix and have the same length.
    let mut disinct = HashSet::with_capacity(views.len() / 2);
    views.iter().for_each(|&view| {
        // SAFETY: we're doing bitwise manipulations. Did you expect that to be safe??
        let view_u128: u128 = unsafe { std::mem::transmute(view) };
        let len_and_prefix = view_u128 as u64;
        disinct.insert(len_and_prefix);
    });

    disinct
        .len()
        .try_into()
        .vortex_expect("distinct count must fit in u32")
}

impl CompressorStats for StringStats {
    type ArrayType = VarBinViewArray;

    fn generate_opts(input: &Self::ArrayType, opts: GenerateStatsOptions) -> Self {
        let null_count = input
            .validity()
            .null_count(input.len())
            .vortex_expect("null_count");
        let value_count = input.len() - null_count;
        let estimated_distinct = if opts.count_distinct_values {
            estimate_distinct_count(input)
        } else {
            u32::MAX
        };

        Self {
            src: input.clone(),
            value_count: value_count.try_into().vortex_expect("value_count"),
            // null_count: null_count.try_into().vortex_expect("null_count"),
            estimated_distinct_count: estimated_distinct,
        }
    }

    fn source(&self) -> &Self::ArrayType {
        &self.src
    }

    fn sample_opts(&self, sample_size: u16, sample_count: u16, opts: GenerateStatsOptions) -> Self {
        let sampled = sample(self.src.clone(), sample_size, sample_count)
            .into_varbinview()
            .vortex_expect("varbinview");

        Self::generate_opts(&sampled, opts)
    }
}

pub struct StringCompressor;

impl Compressor for StringCompressor {
    type ArrayType = VarBinViewArray;
    type SchemeType = dyn StringScheme;
    type StatsType = StringStats;

    fn schemes() -> &'static [&'static Self::SchemeType] {
        &[&UncompressedScheme, &DictScheme, &FSSTScheme]
    }

    fn default_scheme() -> &'static Self::SchemeType {
        &UncompressedScheme
    }

    fn dict_scheme_code() -> StringCode {
        DICT_SCHEME
    }
}

pub trait StringScheme: Scheme<StatsType = StringStats, CodeType = StringCode> {}

impl<T> StringScheme for T where T: Scheme<StatsType = StringStats, CodeType = StringCode> {}

#[derive(Debug, Copy, Clone)]
pub struct UncompressedScheme;

#[derive(Debug, Copy, Clone)]
pub struct DictScheme;

#[derive(Debug, Copy, Clone)]
pub struct FSSTScheme;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct StringCode(u8);

const UNCOMPRESSED_SCHEME: StringCode = StringCode(0);
const DICT_SCHEME: StringCode = StringCode(1);
const FSST_SCHEME: StringCode = StringCode(2);

impl Scheme for UncompressedScheme {
    type StatsType = StringStats;
    type CodeType = StringCode;

    fn code(&self) -> StringCode {
        UNCOMPRESSED_SCHEME
    }

    fn expected_compression_ratio(
        &self,
        _stats: &Self::StatsType,
        _is_sample: bool,
        _allowed_cascading: usize,
        _excludes: &[StringCode],
    ) -> VortexResult<f64> {
        Ok(1.0)
    }

    fn compress(
        &self,
        stats: &Self::StatsType,
        _is_sample: bool,
        _allowed_cascading: usize,
        _excludes: &[StringCode],
    ) -> VortexResult<Array> {
        Ok(stats.source().clone().into_array())
    }
}

impl Scheme for DictScheme {
    type StatsType = StringStats;
    type CodeType = StringCode;

    fn code(&self) -> StringCode {
        DICT_SCHEME
    }

    fn expected_compression_ratio(
        &self,
        stats: &Self::StatsType,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[StringCode],
    ) -> VortexResult<f64> {
        // If we don't have a sufficiently high number of distinct values, do not attempt Dict.
        if stats.estimated_distinct_count > stats.value_count / 2 {
            return Ok(0.0);
        }

        // If array is all null, do not attempt dict.
        if stats.value_count == 0 {
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
        is_sample: bool,
        allowed_cascading: usize,
        _excludes: &[StringCode],
    ) -> VortexResult<Array> {
        let dict = dict_encode(&stats.source().clone().into_array())?;

        // If we are not allowed to cascade, do not attempt codes or values compression.
        if allowed_cascading == 0 {
            return Ok(dict.into_array());
        }

        // Find best compressor for codes and values separately
        let downscaled_codes = downscale_integer_array(dict.codes())?.into_primitive()?;
        let compressed_codes = IntCompressor::compress(
            &downscaled_codes,
            is_sample,
            allowed_cascading - 1,
            &[crate::integer::DictScheme.code()],
        )?;

        // Attempt to compress the values with non-Dict compression.
        // Currently this will only be FSST.
        let compressed_values = StringCompressor::compress(
            &dict.values().into_varbinview()?,
            is_sample,
            allowed_cascading - 1,
            &[DictScheme.code()],
        )?;

        Ok(DictArray::try_new(compressed_codes, compressed_values)?.into_array())
    }
}

impl Scheme for FSSTScheme {
    type StatsType = StringStats;
    type CodeType = StringCode;

    fn code(&self) -> StringCode {
        FSST_SCHEME
    }

    fn compress(
        &self,
        stats: &Self::StatsType,
        _is_sample: bool,
        _allowed_cascading: usize,
        _excludes: &[StringCode],
    ) -> VortexResult<Array> {
        let compressor = fsst_train_compressor(&stats.src.clone().into_array())?;
        let fsst = fsst_compress(&stats.src.clone().into_array(), &compressor)?;

        Ok(fsst.into_array())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::VarBinViewArray;
    use vortex_dtype::{DType, Nullability};

    use crate::string::StringCompressor;
    use crate::Compressor;

    #[test]
    fn test_strings() {
        let mut strings = Vec::new();
        for _ in 0..1024 {
            strings.push(Some("hello-world-1234"));
        }
        for _ in 0..1024 {
            strings.push(Some("hello-world-56789"));
        }
        let strings = VarBinViewArray::from_iter(strings, DType::Utf8(Nullability::NonNullable));

        println!("original array: {}", strings.tree_display());

        let compressed = StringCompressor::compress(&strings, false, 3, &[]).unwrap();

        println!("compression tree: {}", compressed.tree_display());
    }
}
