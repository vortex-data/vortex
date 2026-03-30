// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Canonical array compression implementation.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::CanonicalValidity;
use vortex_array::DynArray;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::ToCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::arrays::listview::list_from_list_view;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::extension::datetime::TemporalMetadata;
use vortex_array::vtable::ValidityHelper;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

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
    pub fn compress(&self, array: &ArrayRef) -> VortexResult<ArrayRef> {
        // Canonicalize the array
        // TODO(joe): receive `ctx` and use it.
        let canonical = array
            .clone()
            .execute::<CanonicalValidity>(&mut LEGACY_SESSION.create_execution_ctx())?
            .0;

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
                    .iter_unmasked_fields()
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
                if list_view_array.is_zero_copy_to_list() || list_view_array.elements().is_empty() {
                    // Offsets are already monotonic and non-overlapping, so we
                    // can drop the sizes array and compress as a ListArray.
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
                if let Ok(temporal_array) = TemporalArray::try_from(ext_array.clone().into_array())
                    && let TemporalMetadata::Timestamp(..) = temporal_array.temporal_metadata()
                {
                    let mut ctx = LEGACY_SESSION.create_execution_ctx();
                    if is_constant(&ext_array.clone().into_array(), &mut ctx)? {
                        return Ok(ConstantArray::new(
                            temporal_array.as_ref().scalar_at(0)?,
                            ext_array.len(),
                        )
                        .into_array());
                    }
                    return compress_temporal(self, temporal_array);
                }

                // Compress the underlying storage array.
                let compressed_storage = self.compress(ext_array.storage_array())?;

                Ok(
                    ExtensionArray::new(ext_array.ext_dtype().clone(), compressed_storage)
                        .into_array(),
                )
            }
            Canonical::Variant(_) => {
                vortex_bail!("Variant arrays can not be compressed")
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
    use vortex_array::DynArray;
    use vortex_array::IntoArray;
    use vortex_array::arrays::List;
    use vortex_array::arrays::ListView;
    use vortex_array::arrays::ListViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::BtrBlocksCompressor;

    #[rstest]
    #[case::zctl(
        unsafe {
            ListViewArray::new_unchecked(
                buffer![1i32, 2, 3, 4, 5].into_array(),
                buffer![0i32, 3].into_array(),
                buffer![3i32, 2].into_array(),
                Validity::NonNullable,
            ).with_zero_copy_to_list(true)
        },
        true,
    )]
    #[case::overlapping(
        ListViewArray::new(
            buffer![1i32, 2, 3].into_array(),
            buffer![0i32, 0, 0].into_array(),
            buffer![3i32, 3, 3].into_array(),
            Validity::NonNullable,
        ),
        false,
    )]
    fn listview_compress_roundtrip(
        #[case] input: ListViewArray,
        #[case] expect_list: bool,
    ) -> VortexResult<()> {
        let array_ref = input.clone().into_array();
        let result = BtrBlocksCompressor::default().compress(&array_ref)?;
        if expect_list {
            assert!(result.as_opt::<List>().is_some());
        } else {
            assert!(result.as_opt::<ListView>().is_some());
        }
        assert_arrays_eq!(result, input);
        Ok(())
    }
}
