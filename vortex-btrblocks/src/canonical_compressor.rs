// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Canonical array compression implementation.

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::arrays::list_from_list_view;
use vortex_array::compute::Cost;
use vortex_array::compute::IsConstantOpts;
use vortex_array::compute::is_constant_opts;
use vortex_array::compute::sum;
use vortex_array::vtable::ValidityHelper;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::datetime::TemporalMetadata;
use vortex_error::VortexResult;

use crate::BtrBlocksCompressorBuilder;
use crate::CompressorContext;
use crate::CompressorExt;
use crate::Excludes;
use crate::FloatCompressor;
use crate::IntCode;
use crate::IntCompressor;
use crate::StringCompressor;
use crate::compressor::decimal::compress_decimal;
use crate::compressor::float::FloatScheme;
use crate::compressor::integer::IntegerScheme;
use crate::compressor::string::StringScheme;
use crate::compressor::temporal::compress_temporal;
use crate::sample::sample;
use crate::sample::sample_count_approx_one_percent;
use crate::stats::SAMPLE_SIZE;

/// Maximum ratio of expanded (List) element count to shared (ListView) element count
/// below which we prefer List encoding over ListView.
const MAX_LIST_EXPANSION_RATIO: f64 = 1.5;

/// Trait for compressors that can compress canonical arrays.
///
/// Provides access to configured compression schemes and the ability to
/// compress canonical arrays recursively.
pub trait CanonicalCompressor {
    /// Compresses a canonical array with the specified options.
    fn compress_canonical(
        &self,
        array: Canonical,
        ctx: CompressorContext,
        excludes: Excludes,
    ) -> VortexResult<ArrayRef>;

    /// Returns the enabled integer compression schemes.
    fn int_schemes(&self) -> &[&'static dyn IntegerScheme];

    /// Returns the enabled float compression schemes.
    fn float_schemes(&self) -> &[&'static dyn FloatScheme];

    /// Returns the enabled string compression schemes.
    fn string_schemes(&self) -> &[&'static dyn StringScheme];
}

/// The main compressor type implementing BtrBlocks-inspired compression.
///
/// This compressor applies adaptive compression schemes to arrays based on their data types
/// and characteristics. It recursively compresses nested structures like structs and lists,
/// and chooses optimal compression schemes for primitive types.
///
/// The compressor works by:
/// 1. Canonicalizing input arrays to a standard representation
/// 2. Analyzing data characteristics to choose optimal compression schemes
/// 3. Recursively compressing nested structures
/// 4. Applying type-specific compression for primitives, strings, and temporal data
///
/// Use [`BtrBlocksCompressorBuilder`] to configure which compression schemes are enabled.
///
/// # Examples
///
/// ```rust
/// use vortex_btrblocks::{BtrBlocksCompressor, BtrBlocksCompressorBuilder, IntCode};
///
/// // Default compressor - all schemes allowed
/// let compressor = BtrBlocksCompressor::default();
///
/// // Exclude specific schemes using the builder
/// let compressor = BtrBlocksCompressorBuilder::default()
///     .exclude_int([IntCode::Dict])
///     .build();
/// ```
#[derive(Clone)]
pub struct BtrBlocksCompressor {
    /// Integer compressor with configured schemes.
    pub int_schemes: Vec<&'static dyn IntegerScheme>,

    /// Float compressor with configured schemes.
    pub float_schemes: Vec<&'static dyn FloatScheme>,

    /// String compressor with configured schemes.
    pub string_schemes: Vec<&'static dyn StringScheme>,
}

impl Default for BtrBlocksCompressor {
    fn default() -> Self {
        BtrBlocksCompressorBuilder::default().build()
    }
}

impl BtrBlocksCompressor {
    /// Compresses an array using BtrBlocks-inspired compression.
    ///
    /// First canonicalizes and compacts the array, then applies optimal compression schemes.
    pub fn compress(&self, array: &dyn Array) -> VortexResult<ArrayRef> {
        // Canonicalize the array
        let canonical = array.to_canonical()?;

        // Compact it, removing any wasted space before we attempt to compress it
        let compact = canonical.compact()?;

        self.compress_canonical(compact, CompressorContext::default(), Excludes::none())
    }

    pub(crate) fn integer_compressor(&self) -> IntCompressor<'_> {
        IntCompressor {
            btr_blocks_compressor: self,
        }
    }

    pub(crate) fn float_compressor(&self) -> FloatCompressor<'_> {
        FloatCompressor {
            btr_blocks_compressor: self,
        }
    }

    pub(crate) fn string_compressor(&self) -> StringCompressor<'_> {
        StringCompressor {
            btr_blocks_compressor: self,
        }
    }

    /// Compresses a [`ListArray`] by narrowing offsets and recursively compressing elements.
    fn compress_list_array(
        &self,
        list_array: ListArray,
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        // Reset the offsets to remove garbage data that might prevent us from narrowing our
        // offsets (there could be a large amount of trailing garbage data that the current
        // views do not reference at all).
        let list_array = list_array.reset_offsets(true)?;

        let compressed_elems = self.compress(list_array.elements())?;

        // Note that since the type of our offsets are not encoded in our `DType`, and since
        // we guarantee above that all elements are referenced by offsets, we may narrow the
        // widths.
        let compressed_offsets = self.compress_canonical(
            Canonical::Primitive(list_array.offsets().to_primitive().narrow()?),
            ctx,
            Excludes::from(&[IntCode::Dict]),
        )?;

        Ok(ListArray::try_new(
            compressed_elems,
            compressed_offsets,
            list_array.validity().clone(),
        )?
        .into_array())
    }

    /// Compresses a [`ListViewArray`] by narrowing offsets/sizes and recursively compressing
    /// elements.
    fn compress_list_view_array(
        &self,
        list_view: ListViewArray,
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let compressed_elems = self.compress(list_view.elements())?;
        let compressed_offsets = self.compress_canonical(
            Canonical::Primitive(list_view.offsets().to_primitive().narrow()?),
            ctx,
            Excludes::none(),
        )?;
        let compressed_sizes = self.compress_canonical(
            Canonical::Primitive(list_view.sizes().to_primitive().narrow()?),
            ctx,
            Excludes::none(),
        )?;
        Ok(ListViewArray::try_new(
            compressed_elems,
            compressed_offsets,
            compressed_sizes,
            list_view.validity().clone(),
        )?
        .into_array())
    }
}

impl CanonicalCompressor for BtrBlocksCompressor {
    /// Compresses a canonical array by dispatching to type-specific compressors.
    ///
    /// Recursively compresses nested structures and applies optimal schemes for each data type.
    fn compress_canonical(
        &self,
        array: Canonical,
        ctx: CompressorContext,
        excludes: Excludes,
    ) -> VortexResult<ArrayRef> {
        match array {
            Canonical::Null(null_array) => Ok(null_array.into_array()),
            // TODO(aduffy): Sparse, other bool compressors.
            Canonical::Bool(bool_array) => Ok(bool_array.into_array()),
            Canonical::Primitive(primitive) => {
                if primitive.ptype().is_int() {
                    self.integer_compressor()
                        .compress(self, &primitive, ctx, excludes.int)
                } else {
                    self.float_compressor()
                        .compress(self, &primitive, ctx, excludes.float)
                }
            }
            Canonical::Decimal(decimal) => compress_decimal(self, &decimal),
            Canonical::Struct(struct_array) => {
                let fields = struct_array
                    .unmasked_fields()
                    .iter()
                    .map(|field| self.compress(field))
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(StructArray::try_new(
                    struct_array.names().clone(),
                    fields,
                    struct_array.len(),
                    struct_array.validity().clone(),
                )?
                .into_array())
            }
            Canonical::List(list_view_array) => {
                let elements_len = list_view_array.elements().len();
                if list_view_array.is_zero_copy_to_list() || elements_len == 0 {
                    // We can avoid the sizes array.
                    let list_array = list_from_list_view(list_view_array)?;
                    return self.compress_list_array(list_array, ctx);
                }

                // Sample the sizes to estimate the total expanded element
                // count, then decide List vs ListView with the expansion
                // threshold.
                let sampled_sizes = sample(
                    list_view_array.sizes(),
                    SAMPLE_SIZE,
                    sample_count_approx_one_percent(list_view_array.len()),
                );
                let sampled_sum = sum(&*sampled_sizes)?
                    .as_primitive()
                    .as_::<usize>()
                    .unwrap_or(0);

                let estimated_expanded_elements_len =
                    sampled_sum * list_view_array.len() / sampled_sizes.len();

                if estimated_expanded_elements_len as f64
                    <= elements_len as f64 * MAX_LIST_EXPANSION_RATIO
                {
                    let list_array = list_from_list_view(list_view_array)?;
                    self.compress_list_array(list_array, ctx)
                } else {
                    self.compress_list_view_array(list_view_array, ctx)
                }
            }
            Canonical::FixedSizeList(fsl_array) => {
                let compressed_elems = self.compress(fsl_array.elements())?;

                Ok(FixedSizeListArray::try_new(
                    compressed_elems,
                    fsl_array.list_size(),
                    fsl_array.validity().clone(),
                    fsl_array.len(),
                )?
                .into_array())
            }
            Canonical::VarBinView(strings) => {
                if strings
                    .dtype()
                    .eq_ignore_nullability(&DType::Utf8(Nullability::NonNullable))
                {
                    self.string_compressor()
                        .compress(self, &strings, ctx, excludes.string)
                } else {
                    // Binary arrays do not compress
                    Ok(strings.into_array())
                }
            }
            Canonical::Extension(ext_array) => {
                // We compress Timestamp-level arrays with DateTimeParts compression
                if let Ok(temporal_array) = TemporalArray::try_from(ext_array.to_array())
                    && let TemporalMetadata::Timestamp(..) = temporal_array.temporal_metadata()
                {
                    if is_constant_opts(
                        temporal_array.as_ref(),
                        &IsConstantOpts {
                            cost: Cost::Canonicalize,
                        },
                    )?
                    .unwrap_or_default()
                    {
                        return Ok(ConstantArray::new(
                            temporal_array.as_ref().scalar_at(0)?,
                            ext_array.len(),
                        )
                        .into_array());
                    }
                    return compress_temporal(self, temporal_array);
                }

                // Compress the underlying storage array.
                let compressed_storage = self.compress(ext_array.storage())?;

                Ok(
                    ExtensionArray::new(ext_array.ext_dtype().clone(), compressed_storage)
                        .into_array(),
                )
            }
        }
    }

    fn int_schemes(&self) -> &[&'static dyn IntegerScheme] {
        &self.int_schemes
    }

    fn float_schemes(&self) -> &[&'static dyn FloatScheme] {
        &self.float_schemes
    }

    fn string_schemes(&self) -> &[&'static dyn StringScheme] {
        &self.string_schemes
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::Array;
    use vortex_array::IntoArray;
    use vortex_array::arrays::ListVTable;
    use vortex_array::arrays::ListViewArray;
    use vortex_array::arrays::ListViewVTable;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::BtrBlocksCompressor;

    /// ZCTL: [[1,2,3], [4,5], [6,7,8,9]]. Monotonic offsets, no overlap.
    fn zctl_listview() -> ListViewArray {
        let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9].into_array();
        let offsets = buffer![0i32, 3, 5].into_array();
        let sizes = buffer![3i32, 2, 4].into_array();
        unsafe {
            ListViewArray::new_unchecked(elements, offsets, sizes, Validity::NonNullable)
                .with_zero_copy_to_list(true)
        }
    }

    /// Non-ZCTL, low duplication: [[7,8,9], [1,2,3], [4,5,6]]. Unsorted but disjoint.
    fn non_zctl_low_dup_listview() -> ListViewArray {
        let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9].into_array();
        let offsets = buffer![6i32, 0, 3].into_array();
        let sizes = buffer![3i32, 3, 3].into_array();
        ListViewArray::new(elements, offsets, sizes, Validity::NonNullable)
    }

    /// Non-ZCTL, high duplication: [[1,2,3]] x 4.
    fn non_zctl_high_dup_listview() -> ListViewArray {
        let elements = buffer![1i32, 2, 3].into_array();
        let offsets = buffer![0i32, 0, 0, 0].into_array();
        let sizes = buffer![3i32, 3, 3, 3].into_array();
        ListViewArray::new(elements, offsets, sizes, Validity::NonNullable)
    }

    /// Nullable with overlap: [[1,2,3], null, [1,2,3], [1,2,3]].
    fn nullable_overlap_listview() -> ListViewArray {
        let elements = buffer![1i32, 2, 3].into_array();
        let offsets = buffer![0i32, 0, 0, 0].into_array();
        let sizes = buffer![3i32, 0, 3, 3].into_array();
        let validity = Validity::from_iter([true, false, true, true]);
        ListViewArray::new(elements, offsets, sizes, validity)
    }

    /// Tests that each ListView variant compresses to the expected encoding and roundtrips.
    #[rstest]
    #[case::zctl(zctl_listview(), true)]
    #[case::non_zctl_low_dup(non_zctl_low_dup_listview(), true)]
    #[case::non_zctl_high_dup(non_zctl_high_dup_listview(), false)]
    #[case::nullable_overlap(nullable_overlap_listview(), false)]
    fn list_view_compress_roundtrip(
        #[case] input: ListViewArray,
        #[case] expect_list: bool,
    ) -> VortexResult<()> {
        let compressor = BtrBlocksCompressor::default();
        let result = compressor.compress(input.as_ref())?;

        if expect_list {
            assert!(
                result.as_opt::<ListVTable>().is_some(),
                "Expected ListArray, got: {}",
                result.encoding_id()
            );
        } else {
            assert!(
                result.as_opt::<ListViewVTable>().is_some(),
                "Expected ListViewArray, got: {}",
                result.encoding_id()
            );
        }

        assert_arrays_eq!(result, input);
        Ok(())
    }
}
