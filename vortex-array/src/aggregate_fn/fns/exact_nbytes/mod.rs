// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The exact number of bytes an array needs in its canonical form.
//!
//! [`ArrayRef::nbytes`] sums the length of every buffer reachable from an array. That is cheap,
//! but it over-counts whenever buffers are shared or only partly referenced: a slice of a
//! [`VarBinViewArray`] keeps every data buffer of its parent, so ten 8K-row slices of one string
//! column each report the whole column's string data. [`ExactNBytes`] instead counts only the
//! bytes the canonical form actually references.

#[cfg(test)]
mod tests;

use std::mem::size_of;
use std::ops::Range;

use num_traits::AsPrimitive;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::Canonical;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::EmptyOptions;
use crate::array::ArrayView;
use crate::arrays::BoolArray;
use crate::arrays::Constant;
use crate::arrays::DecimalArray;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::UnionArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::Variant;
use crate::arrays::VariantArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::arrays::map::MapArraySlotsExt;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::arrays::struct_::StructArrayExt;
use crate::arrays::union::UnionArrayExt;
use crate::arrays::union::UnionArraySlotsExt;
use crate::arrays::varbinview::BinaryView;
use crate::arrays::variant::VariantArraySlotsExt;
use crate::dtype::DType;
use crate::dtype::DecimalType;
use crate::dtype::Nullability::NonNullable;
use crate::dtype::PType;
use crate::match_each_unsigned_integer_ptype;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// Return the exact number of bytes `array` needs in its canonical form.
///
/// See [`ExactNBytes`] for what is counted.
pub fn exact_nbytes(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<u64> {
    let mut acc = Accumulator::try_new(ExactNBytes, EmptyOptions, array.dtype().clone())?;
    acc.accumulate(array, ctx)?;
    acc.finish()?
        .as_primitive()
        .typed_value::<u64>()
        .ok_or_else(|| vortex_err!("exact_nbytes result should not be null"))
}

/// The exact number of bytes an array needs in its canonical form.
///
/// Unlike [`ArrayRef::nbytes`], which sums every buffer reachable from the array, this counts
/// only bytes the canonical form references:
///
/// - Fixed-width slots are counted for every row, valid or not, at the array's own physical
///   width (primitive values, decimal values, binary views, list offsets and sizes).
/// - Bit-packed values and validity bitmaps count `len.div_ceil(8)` bytes. Validity that is not
///   backed by an array (non-nullable, all valid, all invalid) counts nothing.
/// - Out-of-line string and binary data counts only the byte ranges referenced by valid,
///   non-inlined views. Ranges that overlap, including ranges into the same allocation reached
///   through different buffers, are counted once.
/// - List elements count only the elements referenced by valid lists, counting overlapping
///   lists once.
/// - Constant arrays count what canonicalizing them would allocate: one slot per row, and a
///   single copy of an out-of-line string or binary value.
/// - Variant core storage is opaque. A canonical variant child is measured recursively; any other
///   encoding falls back to summing its own buffers and measuring its children.
///
/// The result is a non-null `u64` for every input dtype.
#[derive(Clone, Debug)]
pub struct ExactNBytes;

impl AggregateFnVTable for ExactNBytes {
    type Options = EmptyOptions;
    type Partial = u64;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.exact_nbytes");
        *ID
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn return_dtype(&self, _options: &Self::Options, _input_dtype: &DType) -> Option<DType> {
        Some(DType::Primitive(PType::U64, NonNullable))
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
    ) -> VortexResult<Self::Partial> {
        Ok(0)
    }

    fn partial_from_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        scalar: &Scalar,
    ) -> VortexResult<Self::Partial> {
        scalar
            .as_primitive()
            .typed_value::<u64>()
            .ok_or_else(|| vortex_err!("exact_nbytes partial should not be null"))
    }

    fn merge_partials(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        first: Self::Partial,
        second: Self::Partial,
    ) -> VortexResult<Self::Partial> {
        checked_sum(first, second)
    }

    fn to_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        Ok(Scalar::primitive(*partial, NonNullable))
    }

    #[inline]
    fn is_saturated(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        _partial: &Self::Partial,
    ) -> bool {
        false
    }

    fn accumulate(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let size = match batch {
            Columnar::Canonical(canonical) => canonical_exact_nbytes(canonical, ctx)?,
            Columnar::Constant(constant) => constant_exact_nbytes(constant.as_view(), ctx)?,
        };
        *partial = checked_sum(*partial, size)?;
        Ok(())
    }

    fn finalize(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partials: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        Ok(partials)
    }

    fn finalize_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        self.to_scalar(args, partial)
    }
}

fn canonical_exact_nbytes(canonical: &Canonical, ctx: &mut ExecutionCtx) -> VortexResult<u64> {
    match canonical {
        Canonical::Null(_) => Ok(0),
        Canonical::Bool(array) => bool_exact_nbytes(array),
        Canonical::Primitive(array) => primitive_exact_nbytes(array),
        Canonical::Decimal(array) => decimal_exact_nbytes(array),
        Canonical::VarBinView(array) => varbinview_exact_nbytes(array, ctx),
        Canonical::List(array) => list_view_exact_nbytes(array, ctx),
        Canonical::Map(array) => {
            list_view_exact_nbytes(&array.entries().as_::<ListView>().into_owned(), ctx)
        }
        Canonical::FixedSizeList(array) => fixed_size_list_exact_nbytes(array, ctx),
        Canonical::Struct(array) => struct_exact_nbytes(array, ctx),
        Canonical::Union(array) => union_exact_nbytes(array, ctx),
        Canonical::Extension(array) => extension_exact_nbytes(array, ctx),
        Canonical::Variant(array) => variant_exact_nbytes(array, ctx),
    }
}

fn bool_exact_nbytes(array: &BoolArray) -> VortexResult<u64> {
    let validity = array.as_ref().validity()?;
    checked_sum(
        to_u64(array.len().div_ceil(8))?,
        validity_exact_nbytes(&validity, array.len())?,
    )
}

fn primitive_exact_nbytes(array: &PrimitiveArray) -> VortexResult<u64> {
    let validity = array.as_ref().validity()?;
    checked_sum(
        fixed_width_nbytes(array.len(), array.ptype().byte_width())?,
        validity_exact_nbytes(&validity, array.len())?,
    )
}

fn decimal_exact_nbytes(array: &DecimalArray) -> VortexResult<u64> {
    let validity = array.as_ref().validity()?;
    checked_sum(
        fixed_width_nbytes(array.len(), array.values_type().byte_width())?,
        validity_exact_nbytes(&validity, array.len())?,
    )
}

fn varbinview_exact_nbytes(array: &VarBinViewArray, ctx: &mut ExecutionCtx) -> VortexResult<u64> {
    let len = array.len();
    let validity = array.as_ref().validity()?;
    let views_size = fixed_width_nbytes(len, size_of::<BinaryView>())?;

    let mask = validity.execute_mask(len, ctx)?;
    let data_size = if mask.all_false() {
        0
    } else {
        let buffer_starts: Vec<usize> = array
            .data_buffers()
            .iter()
            .map(|buffer| buffer.as_host().as_ptr() as usize)
            .collect();

        // Referenced data is measured in address space, so the same allocation reached through
        // several buffers, or through overlapping views, is counted once.
        let mut ranges = Vec::new();
        for (idx, view) in array.views().iter().enumerate() {
            if view.is_inlined() || view.is_empty() || !mask.value(idx) {
                continue;
            }
            let view = view.as_view();
            let start = buffer_starts[view.buffer_index as usize] + view.offset as usize;
            ranges.push(start..start + view.size as usize);
        }
        to_u64(covered_len(&merge_ranges(ranges)))?
    };

    checked_sum(
        checked_sum(views_size, data_size)?,
        validity_exact_nbytes(&validity, len)?,
    )
}

fn list_view_exact_nbytes(array: &ListViewArray, ctx: &mut ExecutionCtx) -> VortexResult<u64> {
    let len = array.len();
    let validity = array.as_ref().validity()?;
    let offsets = array.offsets();
    let sizes = array.sizes();
    let views_size = checked_sum(
        fixed_width_nbytes(len, offsets.dtype().as_ptype().byte_width())?,
        fixed_width_nbytes(len, sizes.dtype().as_ptype().byte_width())?,
    )?;

    let elements = array.elements();
    let mask = validity.execute_mask(len, ctx)?;
    let elements_size = if elements.is_empty() || mask.all_false() {
        0
    } else {
        let referenced = referenced_list_elements(offsets, sizes, &mask, ctx)?;
        match referenced.as_slice() {
            [] => 0,
            [range] if *range == (0..elements.len()) => exact_nbytes(elements, ctx)?,
            [range] => exact_nbytes(&elements.slice(range.clone())?, ctx)?,
            ranges => {
                let slices = ranges
                    .iter()
                    .map(|range| (range.start, range.end))
                    .collect();
                let filtered = elements.filter(Mask::from_slices(elements.len(), slices))?;
                exact_nbytes(&filtered, ctx)?
            }
        }
    };

    checked_sum(
        checked_sum(views_size, elements_size)?,
        validity_exact_nbytes(&validity, len)?,
    )
}

/// The sorted, disjoint element ranges referenced by the valid, non-empty lists.
fn referenced_list_elements(
    offsets: &ArrayRef,
    sizes: &ArrayRef,
    validity: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<Range<usize>>> {
    let offsets = offsets.clone().execute::<PrimitiveArray>(ctx)?;
    let sizes = sizes.clone().execute::<PrimitiveArray>(ctx)?;
    // Offsets and sizes are non-negative, so reading them as unsigned is lossless.
    let offsets = offsets.reinterpret_cast(offsets.ptype().to_unsigned());
    let sizes = sizes.reinterpret_cast(sizes.ptype().to_unsigned());

    let mut ranges = Vec::new();
    match_each_unsigned_integer_ptype!(offsets.ptype(), |O| {
        match_each_unsigned_integer_ptype!(sizes.ptype(), |S| {
            let offsets = offsets.as_slice::<O>();
            let sizes = sizes.as_slice::<S>();
            for (idx, (offset, size)) in offsets.iter().zip(sizes).enumerate() {
                let offset: usize = offset.as_();
                let size: usize = size.as_();
                if size > 0 && validity.value(idx) {
                    ranges.push(offset..offset + size);
                }
            }
        })
    });
    Ok(merge_ranges(ranges))
}

fn fixed_size_list_exact_nbytes(
    array: &FixedSizeListArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    let validity = array.as_ref().validity()?;
    checked_sum(
        exact_nbytes(array.elements(), ctx)?,
        validity_exact_nbytes(&validity, array.len())?,
    )
}

fn struct_exact_nbytes(array: &StructArray, ctx: &mut ExecutionCtx) -> VortexResult<u64> {
    let mut size = validity_exact_nbytes(&array.as_ref().validity()?, array.len())?;
    for field in array.iter_unmasked_fields() {
        size = checked_sum(size, exact_nbytes(field, ctx)?)?;
    }
    Ok(size)
}

fn union_exact_nbytes(array: &UnionArray, ctx: &mut ExecutionCtx) -> VortexResult<u64> {
    let mut size = exact_nbytes(array.type_ids(), ctx)?;
    for child in array.iter_children() {
        size = checked_sum(size, exact_nbytes(child, ctx)?)?;
    }
    Ok(size)
}

fn extension_exact_nbytes(array: &ExtensionArray, ctx: &mut ExecutionCtx) -> VortexResult<u64> {
    exact_nbytes(array.storage_array(), ctx)
}

fn variant_exact_nbytes(array: &VariantArray, ctx: &mut ExecutionCtx) -> VortexResult<u64> {
    let mut size = opaque_variant_nbytes(array.core_storage(), ctx)?;
    if let Some(shredded) = array.shredded() {
        size = checked_sum(size, exact_nbytes(shredded, ctx)?)?;
    }
    Ok(size)
}

/// Measure variant core storage without executing it.
///
/// Executing a variant-typed array to canonical can wrap it in a [`VariantArray`] whose core
/// storage is the same array, so going through [`exact_nbytes`] here would not terminate.
fn opaque_variant_nbytes(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<u64> {
    if let Some(variant) = array.as_opt::<Variant>() {
        return variant_exact_nbytes(&variant.into_owned(), ctx);
    }

    let mut size = 0u64;
    for buffer in array.buffers() {
        size = checked_sum(size, to_u64(buffer.len())?)?;
    }
    for child in array.children() {
        let child_size = if matches!(child.dtype(), DType::Variant(_)) {
            opaque_variant_nbytes(&child, ctx)?
        } else {
            exact_nbytes(&child, ctx)?
        };
        size = checked_sum(size, child_size)?;
    }
    Ok(size)
}

fn constant_exact_nbytes(
    array: ArrayView<'_, Constant>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    let len = array.len();
    // Canonicalizing a constant yields all-valid or all-invalid validity, which has no bitmap.
    match array.dtype() {
        DType::Null => Ok(0),
        DType::Bool(_) => to_u64(len.div_ceil(8)),
        DType::Primitive(ptype, _) => fixed_width_nbytes(len, ptype.byte_width()),
        DType::Decimal(decimal_dtype, _) => fixed_width_nbytes(
            len,
            DecimalType::smallest_decimal_value_type(decimal_dtype).byte_width(),
        ),
        DType::Utf8(_) => constant_varbinview_nbytes(
            len,
            array.scalar().as_utf8().value().map(|value| value.len()),
        ),
        DType::Binary(_) => constant_varbinview_nbytes(
            len,
            array.scalar().as_binary().value().map(|value| value.len()),
        ),
        DType::List(..)
        | DType::Map(..)
        | DType::FixedSizeList(..)
        | DType::Struct(..)
        | DType::Union(..)
        | DType::Extension(_)
        | DType::Variant(_) => {
            let canonical = array.array().clone().execute::<Canonical>(ctx)?;
            canonical_exact_nbytes(&canonical, ctx)
        }
    }
}

fn constant_varbinview_nbytes(len: usize, value_len: Option<usize>) -> VortexResult<u64> {
    let views_size = fixed_width_nbytes(len, size_of::<BinaryView>())?;
    // Every view points at the same single copy of an out-of-line value.
    let data_size = match value_len {
        Some(value_len) if len > 0 && value_len > BinaryView::MAX_INLINED_SIZE => {
            to_u64(value_len)?
        }
        _ => 0,
    };
    checked_sum(views_size, data_size)
}

fn validity_exact_nbytes(validity: &Validity, len: usize) -> VortexResult<u64> {
    match validity {
        Validity::Array(_) => to_u64(len.div_ceil(8)),
        Validity::NonNullable | Validity::AllValid | Validity::AllInvalid => Ok(0),
    }
}

/// Sort `ranges` and coalesce the ones that overlap or touch, dropping empty ranges.
fn merge_ranges(mut ranges: Vec<Range<usize>>) -> Vec<Range<usize>> {
    ranges.retain(|range| !range.is_empty());
    if !ranges.is_sorted_by_key(|range| range.start) {
        ranges.sort_unstable_by_key(|range| range.start);
    }

    let mut merged: Vec<Range<usize>> = Vec::with_capacity(ranges.len());
    for range in ranges {
        match merged.last_mut() {
            Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
            _ => merged.push(range),
        }
    }
    merged
}

fn covered_len(ranges: &[Range<usize>]) -> usize {
    ranges.iter().map(|range| range.len()).sum()
}

fn fixed_width_nbytes(len: usize, width: usize) -> VortexResult<u64> {
    to_u64(len)?
        .checked_mul(to_u64(width)?)
        .ok_or_else(|| vortex_err!("exact nbytes overflowed u64"))
}

fn checked_sum(first: u64, second: u64) -> VortexResult<u64> {
    first
        .checked_add(second)
        .ok_or_else(|| vortex_err!("exact nbytes overflowed u64"))
}

fn to_u64(value: usize) -> VortexResult<u64> {
    u64::try_from(value).map_err(|e| vortex_err!("Failed to convert {value} to u64: {e}"))
}
