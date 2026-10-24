// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ExecutionCtx;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::arrays::listview::ListViewRebuildMode;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::IntegerPType;
use vortex_array::match_each_integer_ptype;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use super::UncompressedSizeOpts;
use super::checked_len_mul;
use super::packed_bit_buffer_size_in_bytes;
use super::uncompressed_size_in_bytes_u64;
use super::validity_uncompressed_size_in_bytes;

pub(super) fn list_view_uncompressed_size_in_bytes(
    array: &ListViewArray,
    opts: UncompressedSizeOpts,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    let validity = array
        .as_ref()
        .validity()?
        .execute_mask(array.as_ref().len(), ctx)?;

    let mut size = if array.is_empty() {
        0
    } else if opts.is_exact() {
        referenced_elements_size(array, &validity, opts, ctx)?
    } else {
        uncompressed_size_in_bytes_u64(array.elements(), opts, ctx)?
    };

    let view_buffer_size = u64::try_from(array.len())
        .map_err(|e| vortex_err!("Failed to convert list array length to u64: {e}"))?
        .checked_mul(8)
        .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))?;

    // ListView stores both offsets and sizes as u64 view buffers.
    size = size
        .checked_add(view_buffer_size)
        .and_then(|size| size.checked_add(view_buffer_size))
        .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))?;
    size = size
        .checked_add(validity_uncompressed_size_in_bytes(validity)?)
        .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))?;

    Ok(size)
}

/// The elements the valid lists use, in the range `[start, end)` of the elements array, and how
/// many times they are used in total.
struct ReferencedElements {
    start: usize,
    end: usize,
    count: usize,
}

/// The size of the elements the valid lists use, each counted as many times as a list uses it,
/// found by scanning the offsets and sizes rather than rebuilding the array.
///
/// Lists that together tile one contiguous run of elements, as the lists of any array built from
/// Arrow offsets do, are measured as a slice of that run. Otherwise fixed-width elements are sized
/// by count, and only overlapping or gapped lists of variable-width elements are rebuilt.
fn referenced_elements_size(
    array: &ListViewArray,
    validity: &Mask,
    opts: UncompressedSizeOpts,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    let offsets = array.offsets().clone().execute::<PrimitiveArray>(ctx)?;
    let sizes = array.sizes().clone().execute::<PrimitiveArray>(ctx)?;
    let referenced = match_each_integer_ptype!(offsets.ptype(), |O| {
        match_each_integer_ptype!(sizes.ptype(), |S| {
            referenced_elements(offsets.as_slice::<O>(), sizes.as_slice::<S>(), validity)
        })
    });
    let Some(referenced) = referenced else {
        return Ok(0);
    };

    let elements = array.elements();
    if referenced.count == referenced.end - referenced.start {
        let used = elements.slice(referenced.start..referenced.end)?;
        return uncompressed_size_in_bytes_u64(&used, opts, ctx);
    }

    let values_size = match elements.dtype() {
        DType::Bool(_) => packed_bit_buffer_size_in_bytes(referenced.count)?,
        DType::Primitive(ptype, _) => {
            checked_len_mul(referenced.count, ptype.byte_width(), "primitive")?
        }
        DType::Decimal(decimal, _) => checked_len_mul(
            referenced.count,
            DecimalType::smallest_decimal_value_type(decimal).byte_width(),
            "decimal",
        )?,
        _ => {
            let rebuilt = array.rebuild(ListViewRebuildMode::MakeExact, ctx)?;
            return uncompressed_size_in_bytes_u64(rebuilt.elements(), opts, ctx);
        }
    };

    // The validity of the used elements, as a rebuild would carry it.
    let element_validity = elements.validity()?.execute_mask(elements.len(), ctx)?;
    let nulls = match &element_validity {
        Mask::AllTrue(_) => 0,
        Mask::AllFalse(_) => referenced.count,
        Mask::Values(_) => match_each_integer_ptype!(offsets.ptype(), |O| {
            match_each_integer_ptype!(sizes.ptype(), |S| {
                referenced_null_count(
                    offsets.as_slice::<O>(),
                    sizes.as_slice::<S>(),
                    validity,
                    &element_validity,
                )
            })
        }),
    };
    let validity_size = if nulls == 0 {
        0
    } else if nulls == referenced.count {
        validity_uncompressed_size_in_bytes(Mask::AllFalse(referenced.count))?
    } else {
        packed_bit_buffer_size_in_bytes(referenced.count)?
    };

    values_size
        .checked_add(validity_size)
        .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))
}

fn for_each_valid_list<O: IntegerPType, S: IntegerPType>(
    offsets: &[O],
    sizes: &[S],
    validity: &Mask,
    mut f: impl FnMut(usize, usize),
) {
    let mut visit = |index: usize| {
        let start: usize = offsets[index].as_();
        let size: usize = sizes[index].as_();
        f(start, size);
    };
    match validity {
        Mask::AllTrue(len) => (0..*len).for_each(&mut visit),
        Mask::AllFalse(_) => {}
        Mask::Values(values) => values.indices().iter().copied().for_each(&mut visit),
    }
}

fn referenced_elements<O: IntegerPType, S: IntegerPType>(
    offsets: &[O],
    sizes: &[S],
    validity: &Mask,
) -> Option<ReferencedElements> {
    let mut referenced: Option<ReferencedElements> = None;
    for_each_valid_list(offsets, sizes, validity, |start, size| {
        let end = start + size;
        let referenced = referenced.get_or_insert(ReferencedElements {
            start,
            end,
            count: 0,
        });
        referenced.start = referenced.start.min(start);
        referenced.end = referenced.end.max(end);
        referenced.count += size;
    });
    referenced
}

fn referenced_null_count<O: IntegerPType, S: IntegerPType>(
    offsets: &[O],
    sizes: &[S],
    validity: &Mask,
    element_validity: &Mask,
) -> usize {
    let mut nulls = 0;
    for_each_valid_list(offsets, sizes, validity, |start, size| {
        nulls += element_validity.slice(start..start + size).false_count();
    });
    nulls
}
