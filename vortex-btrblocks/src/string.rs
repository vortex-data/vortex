use vortex_array::array::VarBinViewArray;
use vortex_array::{Array, IntoArray, IntoArrayVariant};
use vortex_dict::{dict_encode, DictArray};
use vortex_error::{VortexExpect, VortexResult};
use vortex_fsst::{fsst_compress, fsst_train_compressor};

use crate::integer::IntCompressor;
use crate::sample::sample;
use crate::{Compressor, CompressorStats, Scheme};

#[derive(Clone)]
pub struct StringStats {
    src: VarBinViewArray,
}

impl CompressorStats for StringStats {
    type ArrayType = VarBinViewArray;

    fn generate(input: &Self::ArrayType) -> Self {
        Self { src: input.clone() }
    }

    fn source(&self) -> &Self::ArrayType {
        &self.src
    }

    fn sample(&self, sample_size: u16, sample_count: u16) -> Self {
        let sampled = sample(self.src.clone(), sample_size, sample_count)
            .into_varbinview()
            .vortex_expect("varbinview");

        Self::generate(&sampled)
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
}

pub trait StringScheme: Scheme<StatsType = StringStats> {}

impl<T> StringScheme for T where T: Scheme<StatsType = StringStats> {}

#[derive(Debug, Copy, Clone)]
pub struct UncompressedScheme;

#[derive(Debug, Copy, Clone)]
pub struct DictScheme;

#[derive(Debug, Copy, Clone)]
pub struct FSSTScheme;

impl Scheme for UncompressedScheme {
    type StatsType = StringStats;

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

impl Scheme for DictScheme {
    type StatsType = StringStats;

    fn code(&self) -> u8 {
        1
    }

    fn compress(
        &self,
        stats: &Self::StatsType,
        is_sample: bool,
        allowed_cascading: usize,
        _excludes: &[u8],
    ) -> VortexResult<Array> {
        let dict = dict_encode(&stats.source().clone().into_array())?;

        if allowed_cascading == 0 {
            return Ok(dict.into_array());
        }

        // Find best compressor for codes and values separately
        let compressed_codes = IntCompressor::compress(
            &dict.codes().into_primitive()?,
            is_sample,
            allowed_cascading - 1,
            &[crate::integer::DictScheme.code()],
        )?;

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

    fn code(&self) -> u8 {
        2
    }

    fn compress(
        &self,
        stats: &Self::StatsType,
        _is_sample: bool,
        _allowed_cascading: usize,
        _excludes: &[u8],
    ) -> VortexResult<Array> {
        let compressor = fsst_train_compressor(&stats.src.clone().into_array())?;
        let fsst = fsst_compress(&stats.src.clone().into_array(), &compressor)?;

        Ok(fsst.into_array())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::array::VarBinViewArray;
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
            strings.push(Some("hello-world-5678"));
        }
        let strings = VarBinViewArray::from_iter(strings, DType::Utf8(Nullability::NonNullable));

        println!("original array: {}", strings.tree_display());

        let compressed = StringCompressor::compress(&strings, false, 3, &[]).unwrap();

        println!("compression tree: {}", compressed.tree_display());
    }
}
