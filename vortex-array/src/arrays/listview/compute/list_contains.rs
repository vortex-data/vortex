// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_buffer::bit_iterator::BitIndexIterator;
use num_traits::Zero;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::ListView;
use crate::arrays::PrimitiveArray;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::Nullability;
use crate::match_each_integer_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::list_contains::ListContainsListKernel;
use crate::scalar_fn::fns::list_contains::ListContainsOptions;
use crate::scalar_fn::fns::operators::Operator;
use crate::validity::Validity;

/// Membership of a constant needle in each list of a [`ListView`] array.
///
/// The needle is compared against the whole elements array in one pass, and the resulting bits are
/// folded per list using the offsets and sizes.
impl ListContainsListKernel for ListView {
    fn list_contains(
        list: ArrayView<'_, Self>,
        needle: &ArrayRef,
        options: &ListContainsOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(value) = needle.as_constant() else {
            return Ok(None);
        };
        // A null needle is never equal to an element, so every row is null: the generic
        // implementation answers that without touching the elements.
        if value.is_null() {
            return Ok(None);
        }
        let nullability = options.result_nullability(list.dtype(), needle.dtype());

        let elems = list.elements();
        if elems.is_empty() {
            // Must return false when a list is empty (but valid), or null when the list itself is
            // null.
            return list_false_or_null(list, nullability, ctx).map(Some);
        }

        let rhs = ConstantArray::new(value, elems.len());
        let matching_elements =
            Binary::try_new(elems.clone(), rhs.into_array(), Operator::Eq)?.into_array();

        // TODO(ngates): we should execute this into a Columnar and check for constant.
        let matches = matching_elements.execute::<BoolArray>(ctx)?;

        // Under SQL null semantics a list holding a null element answers `null` for a needle that
        // matches none of its other elements. The needle is non-null here, so a null comparison is
        // a null element; fold "any match" and "any null" per list and keep only the decided rows.
        if options.sql_null_semantics {
            let valid = matches.validity()?.execute_mask(matches.len(), ctx)?;
            if !valid.all_true() {
                let valid = valid.to_bit_buffer();
                let any_true = fold_lists(
                    BoolArray::new(&matches.to_bit_buffer() & &valid, Validity::NonNullable),
                    list,
                    ctx,
                )?;
                let any_null =
                    fold_lists(BoolArray::new(!valid, Validity::NonNullable), list, ctx)?;
                let decided = &any_true | &!any_null;
                let validity = list
                    .validity()?
                    .and(Validity::from(decided))?
                    .union_nullability(nullability);
                return Ok(Some(BoolArray::new(any_true, validity).into_array()));
            }
        }

        // Fast path: every comparison agrees.
        if let Some(pred) = matches.as_constant() {
            return match pred.as_bool().value() {
                // All comparisons are invalid (result in `null`), and the needle is not null
                // because we already checked for that above.
                None => {
                    // False, unless the list itself is null in which case we return null.
                    list_false_or_null(list, nullability, ctx).map(Some)
                }
                // No elements match, and all comparisons are valid (result in `false`).
                Some(false) => {
                    // False, but match the nullability to the input list array.
                    Ok(Some(
                        ConstantArray::new(Scalar::bool(false, nullability), list.len())
                            .into_array(),
                    ))
                }
                // All elements match, and all comparisons are valid (result in `true`).
                Some(true) => {
                    // True, unless the list itself is empty or NULL.
                    list_is_not_empty(list, nullability, ctx).map(Some)
                }
            };
        }

        let list_matches = fold_lists(matches, list, ctx)?;

        Ok(Some(
            BoolArray::new(
                list_matches,
                list.validity()?.union_nullability(nullability),
            )
            .into_array(),
        ))
    }
}

/// For each list, whether any set bit of `matches` falls in the list's element range.
fn fold_lists(
    matches: BoolArray,
    list_array: ArrayView<'_, ListView>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BitBuffer> {
    // Get the offsets and sizes as primitive arrays. They are non-negative, so reinterpret to
    // unsigned and dispatch over the 4 unsigned widths each (4x4 instead of 8x8).
    let offsets = list_array
        .offsets()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let offsets = offsets.reinterpret_cast(offsets.ptype().to_unsigned());
    let sizes = list_array.sizes().clone().execute::<PrimitiveArray>(ctx)?;
    let sizes = sizes.reinterpret_cast(sizes.ptype().to_unsigned());

    Ok(match_each_unsigned_integer_ptype!(offsets.ptype(), |O| {
        match_each_unsigned_integer_ptype!(sizes.ptype(), |S| {
            process_matches::<O, S>(matches, list_array.len(), offsets, sizes, ctx)
        })
    }))
}

/// Returns a [`BitBuffer`] where each bit represents if a list contains the scalar, derived from a
/// [`BoolArray`] of matches on the child elements array.
fn process_matches<O, S>(
    matches: BoolArray,
    list_array_len: usize,
    offsets: PrimitiveArray,
    sizes: PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> BitBuffer
where
    O: IntegerPType,
    S: IntegerPType,
{
    let offsets_slice = offsets.as_slice::<O>();
    let sizes_slice = sizes.as_slice::<S>();
    let bits = matches.bit_buffer_view();

    BitBuffer::collect_bool_in(
        list_array_len,
        |i| {
            let offset = offsets_slice[i].as_();
            let size = sizes_slice[i].as_();

            // BitIndexIterator yields indices of true bits only. If `.next()` returns
            // `Some(_)`, at least one element in this list's range matches.
            let mut set_bits = BitIndexIterator::new(bits.inner(), offset, size);
            set_bits.next().is_some()
        },
        ctx.allocator().clone(),
    )
}

/// Returns a `Bool` array with `false` for lists that are valid,
/// or `NULL` if the list itself is null.
fn list_false_or_null(
    list_array: ArrayView<'_, ListView>,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    match list_array.validity()? {
        Validity::NonNullable => {
            // All false.
            Ok(ConstantArray::new(Scalar::bool(false, nullability), list_array.len()).into_array())
        }
        Validity::AllValid => {
            // All false, but nullable.
            Ok(
                ConstantArray::new(Scalar::bool(false, Nullability::Nullable), list_array.len())
                    .into_array(),
            )
        }
        Validity::AllInvalid => {
            // All nulls, must be nullable result.
            Ok(ConstantArray::new(
                Scalar::null(DType::Bool(Nullability::Nullable)),
                list_array.len(),
            )
            .into_array())
        }
        Validity::Array(validity_array) => {
            // Create a new bool array with false, and the provided nulls
            let buffer = BitBuffer::new_unset_in(list_array.len(), ctx.allocator().clone());
            Ok(BoolArray::new(buffer, Validity::Array(validity_array)).into_array())
        }
    }
}

/// Returns a `Bool` array with `true` for lists which are NOT empty, or `false` if they are empty,
/// or `NULL` if the list itself is null.
fn list_is_not_empty(
    list_array: ArrayView<'_, ListView>,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // Short-circuit for all invalid.
    if list_array.validity()?.definitely_all_null() {
        return Ok(ConstantArray::new(
            Scalar::null(DType::Bool(Nullability::Nullable)),
            list_array.len(),
        )
        .into_array());
    }

    let sizes = list_array.sizes().clone().execute::<PrimitiveArray>(ctx)?;
    let buffer = match_each_integer_ptype!(sizes.ptype(), |S| {
        let sizes = sizes.as_slice::<S>();
        BitBuffer::collect_bool_in(
            sizes.len(),
            |idx| sizes[idx] != S::zero(),
            ctx.allocator().clone(),
        )
    });

    // Copy over the validity mask from the input.
    Ok(BoolArray::new(
        buffer,
        list_array.validity()?.union_nullability(nullability),
    )
    .into_array())
}
