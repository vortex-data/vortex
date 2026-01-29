// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::VarBinViewVTable;
use vortex_array::builders::dict::dict_encode;
use vortex_array::compute::is_constant;
use vortex_array::vtable::ValidityHelper;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_fsst::FSSTArray;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_scalar::Scalar;
use vortex_sparse::SparseArray;
use vortex_sparse::SparseVTable;
use vortex_utils::aliases::hash_set::HashSet;

use crate::Compressor;
use crate::CompressorStats;
use crate::GenerateStatsOptions;
use crate::Scheme;
use crate::estimate_compression_ratio_with_sampling;
use crate::integer;
use crate::integer::IntCompressor;
use crate::sample::sample;

/// Array of variable-length byte arrays, and relevant stats for compression.
#[derive(Clone, Debug)]
pub struct StringStats {
    src: VarBinViewArray,
    estimated_distinct_count: u32,
    value_count: u32,
    null_count: u32,
}

/// Estimate the number of distinct strings in the var bin view array.
fn estimate_distinct_count(strings: &VarBinViewArray) -> VortexResult<u32> {
    let views = strings.views();
    // Iterate the views. Two strings which are equal must have the same first 8-bytes.
    // NOTE: there are cases where this performs pessimally, e.g. when we have strings that all
    // share a 4-byte prefix and have the same length.
    let mut distinct = HashSet::with_capacity(views.len() / 2);
    views.iter().for_each(|&view| {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "approximate uniqueness with view prefix"
        )]
        let len_and_prefix = view.as_u128() as u64;
        distinct.insert(len_and_prefix);
    });

    Ok(u32::try_from(distinct.len())?)
}

impl StringStats {
    fn generate_opts_fallible(
        input: &VarBinViewArray,
        opts: GenerateStatsOptions,
    ) -> VortexResult<Self> {
        let null_count = input
            .statistics()
            .compute_null_count()
            .ok_or_else(|| vortex_err!("Failed to compute null_count"))?;
        let value_count = input.len() - null_count;
        let estimated_distinct = if opts.count_distinct_values {
            estimate_distinct_count(input)?
        } else {
            u32::MAX
        };

        Ok(Self {
            src: input.clone(),
            value_count: u32::try_from(value_count)?,
            null_count: u32::try_from(null_count)?,
            estimated_distinct_count: estimated_distinct,
        })
    }
}

impl CompressorStats for StringStats {
    type ArrayVTable = VarBinViewVTable;

    fn generate_opts(input: &VarBinViewArray, opts: GenerateStatsOptions) -> Self {
        Self::generate_opts_fallible(input, opts)
            .vortex_expect("StringStats::generate_opts should not fail")
    }

    fn source(&self) -> &VarBinViewArray {
        &self.src
    }

    fn sample_opts(&self, sample_size: u32, sample_count: u32, opts: GenerateStatsOptions) -> Self {
        let sampled = sample(self.src.as_ref(), sample_size, sample_count).to_varbinview();

        Self::generate_opts(&sampled, opts)
    }
}

/// [`Compressor`] for strings.
pub struct StringCompressor;

impl Compressor for StringCompressor {
    type ArrayVTable = VarBinViewVTable;
    type SchemeType = dyn StringScheme;
    type StatsType = StringStats;

    fn schemes() -> &'static [&'static Self::SchemeType] {
        &[
            &UncompressedScheme,
            &DictScheme,
            &FSSTScheme,
            &ConstantScheme,
            &NullDominated,
        ]
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

#[derive(Debug, Copy, Clone)]
pub struct ConstantScheme;

#[derive(Debug, Copy, Clone)]
pub struct NullDominated;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct StringCode(u8);

const UNCOMPRESSED_SCHEME: StringCode = StringCode(0);
const DICT_SCHEME: StringCode = StringCode(1);
const FSST_SCHEME: StringCode = StringCode(2);
const CONSTANT_SCHEME: StringCode = StringCode(3);

const SPARSE_SCHEME: StringCode = StringCode(4);

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
    ) -> VortexResult<ArrayRef> {
        Ok(stats.source().to_array())
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
    ) -> VortexResult<ArrayRef> {
        let dict = dict_encode(&stats.source().clone().into_array())?;

        // If we are not allowed to cascade, do not attempt codes or values compression.
        if allowed_cascading == 0 {
            return Ok(dict.into_array());
        }

        // Find best compressor for codes and values separately
        let compressed_codes = IntCompressor::compress(
            &dict.codes().to_primitive(),
            is_sample,
            allowed_cascading - 1,
            &[integer::DictScheme.code(), integer::SequenceScheme.code()],
        )?;

        // Attempt to compress the values with non-Dict compression.
        // Currently this will only be FSST.
        let compressed_values = StringCompressor::compress(
            &dict.values().to_varbinview(),
            is_sample,
            allowed_cascading - 1,
            &[DictScheme.code()],
        )?;

        // SAFETY: compressing codes or values does not alter the invariants
        unsafe {
            Ok(
                DictArray::new_unchecked(compressed_codes, compressed_values)
                    .set_all_values_referenced(dict.has_all_values_referenced())
                    .into_array(),
            )
        }
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
        is_sample: bool,
        allowed_cascading: usize,
        _excludes: &[StringCode],
    ) -> VortexResult<ArrayRef> {
        let compressor = fsst_train_compressor(&stats.src);
        let fsst = fsst_compress(&stats.src, &compressor);

        let compressed_original_lengths = IntCompressor::compress(
            &fsst.uncompressed_lengths().to_primitive().narrow()?,
            is_sample,
            allowed_cascading,
            &[],
        )?;

        let compressed_codes_offsets = IntCompressor::compress(
            &fsst.codes().offsets().to_primitive().narrow()?,
            is_sample,
            allowed_cascading,
            &[],
        )?;
        let compressed_codes = VarBinArray::try_new(
            compressed_codes_offsets,
            fsst.codes().bytes().clone(),
            fsst.codes().dtype().clone(),
            fsst.codes().validity().clone(),
        )?;

        let fsst = FSSTArray::try_new(
            fsst.dtype().clone(),
            fsst.symbols().clone(),
            fsst.symbol_lengths().clone(),
            compressed_codes,
            compressed_original_lengths,
        )?;

        Ok(fsst.into_array())
    }
}

impl Scheme for ConstantScheme {
    type StatsType = StringStats;
    type CodeType = StringCode;

    fn code(&self) -> Self::CodeType {
        CONSTANT_SCHEME
    }

    fn is_constant(&self) -> bool {
        true
    }

    fn expected_compression_ratio(
        &self,
        stats: &Self::StatsType,
        is_sample: bool,
        _allowed_cascading: usize,
        _excludes: &[Self::CodeType],
    ) -> VortexResult<f64> {
        if is_sample {
            return Ok(0.0);
        }

        if stats.estimated_distinct_count > 1 || !is_constant(stats.src.as_ref())?.unwrap_or(false)
        {
            return Ok(0.0);
        }

        // Force constant is these cases
        Ok(f64::MAX)
    }

    fn compress(
        &self,
        stats: &Self::StatsType,
        _is_sample: bool,
        _allowed_cascading: usize,
        _excludes: &[Self::CodeType],
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

impl Scheme for NullDominated {
    type StatsType = StringStats;
    type CodeType = StringCode;

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

        // We pass None as we only run this pathway for NULL-dominated string arrays
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
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::builders::ArrayBuilder;
    use vortex_array::builders::VarBinViewBuilder;
    use vortex_array::display::DisplayOptions;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_error::VortexResult;

    use crate::Compressor;
    use crate::MAX_CASCADE;
    use crate::string::StringCompressor;

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

        let compressed = StringCompressor::compress(&strings, false, 3, &[])?;
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

        let compressed = StringCompressor::compress(&strings, false, MAX_CASCADE, &[])?;
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
    use vortex_array::arrays::ConstantVTable;
    use vortex_array::arrays::DictVTable;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_error::VortexResult;
    use vortex_fsst::FSSTVTable;

    use crate::Compressor;
    use crate::string::StringCompressor;

    #[test]
    fn test_constant_compressed() -> VortexResult<()> {
        let strings: Vec<Option<&str>> = vec![Some("constant_value"); 100];
        let array = VarBinViewArray::from_iter(strings, DType::Utf8(Nullability::NonNullable));
        let compressed = StringCompressor::compress(&array, false, 3, &[])?;
        assert!(compressed.is::<ConstantVTable>());
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
        let compressed = StringCompressor::compress(&array, false, 3, &[])?;
        assert!(compressed.is::<DictVTable>());
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
        let compressed = StringCompressor::compress(&array, false, 3, &[])?;
        assert!(compressed.is::<FSSTVTable>());
        Ok(())
    }
}
