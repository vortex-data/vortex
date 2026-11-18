// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Drops the nulls of an Arrow array that no row reads, in the levels that a Vortex dtype declares
//! non-nullable.
//!
//! An Arrow producer can mark any nested level of an array nullable, and a struct or fixed-size
//! list can hold nulls under its null slots in a child that it marks non-nullable. A Vortex dtype
//! declares the nullability of every level, and a non-nullable Vortex array holds no nulls. A slot
//! of a nested array is _masked_ when one of its ancestors is null, and _visible_ when every
//! ancestor is valid. [`drop_masked_nulls`] applies the rule that
//! [`ArrowSession::from_arrow_storage`] states to one level.
//!
//! The _visibility_ of a level is an `Option<&NullBuffer>` that marks its visible slots, where
//! `None` means that every slot is visible. A `Some` always masks at least one slot. A struct
//! passes [`NullBuffer::union`] of its visibility and its nulls on to its fields, which keeps this
//! property. A list passes the visibility of its elements on with [`list_elements_visibility`] or
//! [`fixed_size_list_elements_visibility`], which also give `None` when the elements hold no null
//! that their import could drop. So a level that gets `None` never has a masked null to drop.
//!
//! The canonical import in [`ArrowSession`] applies the rule to each level as it converts it.
//! [`with_declared_nullability`] instead rebuilds a whole Arrow array in the declared form, for an
//! array that an importer receives.
//!
//! [`ArrowSession`]: crate::ArrowSession
//! [`ArrowSession::from_arrow_storage`]: crate::ArrowSession::from_arrow_storage

use std::fmt::Display;
use std::sync::Arc;

use arrow_array::Array;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::FixedSizeListArray;
use arrow_array::GenericListArray;
use arrow_array::OffsetSizeTrait;
use arrow_array::StructArray;
use arrow_array::cast::AsArray;
use arrow_array::make_array;
use arrow_buffer::ArrowNativeType;
use arrow_buffer::BooleanBufferBuilder;
use arrow_buffer::NullBuffer;
use arrow_buffer::OffsetBuffer;
use arrow_data::ArrayData;
use arrow_schema::DataType;
use arrow_schema::FieldRef;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::convert::remove_nulls;

/// Drops the masked `nulls` of a level that is not `nullable`, and returns the nulls that the level
/// keeps.
///
/// `visible` is the [visibility](self) of the level. A nullable level keeps all of its nulls, and a
/// non-nullable level keeps none.
///
/// # Errors
///
/// Returns an error that names `level` if it is not `nullable` and holds a null in a visible slot.
pub(crate) fn drop_masked_nulls(
    nulls: Option<&NullBuffer>,
    nullable: bool,
    visible: Option<&NullBuffer>,
    level: &dyn Display,
) -> VortexResult<Option<NullBuffer>> {
    let Some(nulls) = nulls.filter(|nulls| nulls.null_count() > 0) else {
        return Ok(None);
    };
    if nullable {
        return Ok(Some(nulls.clone()));
    }

    let visible_nulls = match visible {
        Some(visible) => visible.inner() & &!nulls.inner(),
        None => !nulls.inner(),
    };
    if let Some(index) = visible_nulls.set_indices().next() {
        vortex_bail!(
            "{level} must not be null unless an ancestor is null, got a null at index {index}"
        );
    }

    Ok(None)
}

/// Rebuilds `array` with the nullability that `dtype` declares at every nested level, dropping the
/// masked nulls of each level with [`drop_masked_nulls`].
///
/// `visible` is the [visibility](self) of `array`. An extension level rebuilds as its storage. A
/// struct, list, large list, or fixed-size list level that pairs with a level of the same kind in
/// `dtype` is rebuilt recursively, where a struct pairs only when its fields have the declared
/// names in order. Any other level only takes its own declared nullability, so the levels below it
/// keep the nullability that Arrow gives them.
///
/// # Errors
///
/// Returns an error if a level that `dtype` declares non-nullable holds a null in a visible slot.
pub(crate) fn with_declared_nullability(
    array: &ArrowArrayRef,
    dtype: &DType,
    visible: Option<&NullBuffer>,
) -> VortexResult<ArrowArrayRef> {
    if let DType::Extension(ext) = dtype {
        return with_declared_nullability(array, ext.storage_dtype(), visible).map_err(|err| {
            err.with_context(format!("Failed to import Arrow storage as {}", ext.id()))
        });
    }

    let declared_nulls = drop_masked_nulls(array.nulls(), dtype.is_nullable(), visible, dtype)?;
    // The nulls of this level mask its children, including the nulls that the rebuild drops.
    let children_visible = NullBuffer::union(visible, array.nulls());

    match (array.data_type(), dtype) {
        // The storage of an extension type can name other fields than its Arrow layout does, which
        // the importer of the type rejects. Pairing those fields by position would rebuild a field
        // with the nullability of another one, or drop the fields past the declared ones.
        (DataType::Struct(fields), DType::Struct(declared_fields, _))
            if declared_fields
                .names()
                .iter()
                .eq(fields.iter().map(|field| field.name().as_str())) =>
        {
            let array = array.as_struct();
            let mut columns = Vec::with_capacity(array.num_columns());
            let mut rebuilt_fields = Vec::with_capacity(array.num_columns());
            for ((column, field), declared) in array
                .columns()
                .iter()
                .zip(fields.iter())
                .zip(declared_fields.fields())
            {
                let column =
                    with_declared_nullability(column, &declared, children_visible.as_ref())?;
                rebuilt_fields.push(declared_field(field, &column, &declared));
                columns.push(column);
            }
            Ok(Arc::new(StructArray::try_new_with_length(
                rebuilt_fields.into(),
                columns,
                declared_nulls,
                array.len(),
            )?))
        }
        (DataType::List(field), DType::List(element, _)) => list_with_declared_nullability(
            array.as_list::<i32>(),
            field,
            element,
            children_visible.as_ref(),
            declared_nulls,
        ),
        (DataType::LargeList(field), DType::List(element, _)) => list_with_declared_nullability(
            array.as_list::<i64>(),
            field,
            element,
            children_visible.as_ref(),
            declared_nulls,
        ),
        (DataType::FixedSizeList(field, size), DType::FixedSizeList(element, ..)) => {
            let list = array.as_fixed_size_list();
            let elements_visible = fixed_size_list_elements_visibility(
                children_visible.as_ref(),
                size.as_usize(),
                list.values().as_ref(),
                element.is_nullable(),
            );
            let values =
                with_declared_nullability(list.values(), element, elements_visible.as_ref())?;
            Ok(Arc::new(FixedSizeListArray::try_new_with_length(
                declared_field(field, &values, element),
                *size,
                values,
                declared_nulls,
                list.len(),
            )?))
        }
        _ if declared_nulls.is_none() && array.null_count() > 0 => {
            Ok(make_array(remove_nulls(array.to_data())?))
        }
        _ => Ok(ArrowArrayRef::clone(array)),
    }
}

/// Rebuilds the elements of `list` with the nullability of `element`, keeping the offsets.
///
/// `rows_visible` marks the visible lists, and the rebuilt list keeps `declared_nulls` as its
/// validity.
fn list_with_declared_nullability<O: OffsetSizeTrait>(
    list: &GenericListArray<O>,
    field: &FieldRef,
    element: &DType,
    rows_visible: Option<&NullBuffer>,
    declared_nulls: Option<NullBuffer>,
) -> VortexResult<ArrowArrayRef> {
    let elements_visible = list_elements_visibility(
        list.offsets(),
        rows_visible,
        list.values().as_ref(),
        element.is_nullable(),
    );
    let values = with_declared_nullability(list.values(), element, elements_visible.as_ref())?;
    Ok(Arc::new(GenericListArray::<O>::try_new(
        declared_field(field, &values, element),
        list.offsets().clone(),
        values,
        declared_nulls,
    )?))
}

/// The [visibility](self) of the `values` of a list with `offsets`, for their import at a level
/// that is `element_nullable`.
///
/// `rows_visible` marks the lists that are visible and valid, and an element is visible when one of
/// them contains it. A sliced list can leave elements outside the range of every list. No list
/// reads them, so they are masked.
pub(crate) fn list_elements_visibility<O: OffsetSizeTrait>(
    offsets: &OffsetBuffer<O>,
    rows_visible: Option<&NullBuffer>,
    values: &dyn Array,
    element_nullable: bool,
) -> Option<NullBuffer> {
    let len = values.len();
    let first = offsets[0].as_usize();
    let last = offsets[offsets.len() - 1].as_usize();
    if rows_visible.is_none() && first == 0 && last == len {
        return None;
    }
    if !reads_visibility(values, element_nullable) {
        return None;
    }

    let mut visible = BooleanBufferBuilder::new(len);
    visible.append_n(first, false);
    for (row, bounds) in offsets.windows(2).enumerate() {
        let row_visible = rows_visible.is_none_or(|rows| rows.is_valid(row));
        visible.append_n(bounds[1].as_usize() - bounds[0].as_usize(), row_visible);
    }
    visible.append_n(len - last, false);

    // Null lists mask no element when they are empty, which is how arrow-rs builders and pyarrow
    // write them.
    let visible = NullBuffer::new(visible.finish());
    (visible.null_count() > 0).then_some(visible)
}

/// The [visibility](self) of the `values` of fixed-size lists of `size`, for their import at a
/// level that is `element_nullable`.
///
/// `rows_visible` marks the lists that are visible and valid, and an element is visible when one of
/// them contains it.
pub(crate) fn fixed_size_list_elements_visibility(
    rows_visible: Option<&NullBuffer>,
    size: usize,
    values: &dyn Array,
    element_nullable: bool,
) -> Option<NullBuffer> {
    let rows_visible = rows_visible.filter(|_| reads_visibility(values, element_nullable))?;

    // NB: arrow-rs `NullBuffer::expand` sets one bit at a time, which dominates the import of
    // vectors under null rows. Appending each list as one run fills whole bytes instead:
    // https://github.com/apache/arrow-rs/blob/59.3.2/arrow-buffer/src/buffer/null.rs#L121-L139
    let mut visible = BooleanBufferBuilder::new(rows_visible.len() * size);
    for row_visible in rows_visible.iter() {
        visible.append_n(size, row_visible);
    }

    Some(NullBuffer::new(visible.finish()))
}

/// Whether the import of `values` at a level that is `nullable` reads the visibility of its slots.
///
/// A non-nullable leaf reads it to check its nulls, and a nested level passes it down when it or a
/// level below it holds a null. Building the visibility of list elements costs a pass over them,
/// which a level without nulls, or a nullable leaf such as the floats of a plain fixed-size list,
/// never needs. For a nested level, this builds and walks every level below it, so the callers
/// first test whether a null row or a slice masks any element at all.
fn reads_visibility(values: &dyn Array, nullable: bool) -> bool {
    if values.data_type().is_nested() {
        holds_nulls(&values.to_data())
    } else {
        !nullable && values.null_count() > 0
    }
}

/// Whether `data` or an array below it holds a null.
fn holds_nulls(data: &ArrayData) -> bool {
    data.null_count() > 0 || data.child_data().iter().any(holds_nulls)
}

/// `field` with the data type of `child` and the nullability that `dtype` declares.
fn declared_field(field: &FieldRef, child: &ArrowArrayRef, dtype: &DType) -> FieldRef {
    Arc::new(
        field
            .as_ref()
            .clone()
            .with_data_type(child.data_type().clone())
            .with_nullable(dtype.is_nullable()),
    )
}
