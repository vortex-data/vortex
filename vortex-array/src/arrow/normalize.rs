// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Normalization of Arrow [`ArrayData`] imported over the C data interface.
//!
//! arrow-rs (at least up to v59) has two conflicting conventions for the `offset` of a `Struct`
//! [`ArrayData`]: [`ArrayData::slice`] pushes the slice into the children *and* records a
//! non-zero parent offset, while `StructArray::from(ArrayData)` applies a non-zero parent offset
//! to the children *again*. `FixedSizeListArray::from(ArrayData)` composes the two by slicing its
//! (possibly struct) child, so importing a sliced `fixed_size_list<struct>` — or any
//! struct-with-offset whose child is also a struct — panics inside [`arrow_array::make_array`]
//! with `assertion failed: end <= self.len()`.
//!
//! [`normalize_array_data`] rewrites such data into an equivalent canonical form where every
//! `Struct` and `FixedSizeList` node carries a zero offset (the slice is materialized into the
//! children's windows), which `make_array` handles correctly.

use arrow_data::ArrayData;
use arrow_schema::DataType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

/// Returns [`ArrayData`] equivalent to `data` that is safe to pass to
/// [`arrow_array::make_array`].
///
/// This is required for any `ArrayData` imported through the Arrow C data interface (e.g. from
/// pyarrow), which may contain sliced `Struct` or `FixedSizeList` nodes at any depth. Data that
/// does not need rewriting is returned unchanged.
pub fn normalize_array_data(data: ArrayData) -> VortexResult<ArrayData> {
    if !needs_normalization(&data) {
        return Ok(data);
    }
    normalize_window(&data, 0, data.len())
}

/// Whether the tree contains a `Struct` or `FixedSizeList` node with a non-zero offset, which
/// `make_array` may slice into a representation it then misinterprets.
fn needs_normalization(data: &ArrayData) -> bool {
    let self_offset = matches!(
        data.data_type(),
        DataType::Struct(_) | DataType::FixedSizeList(..)
    ) && data.offset() != 0;
    self_offset || data.child_data().iter().any(needs_normalization)
}

/// Build an [`ArrayData`] equivalent to `data.slice(rel_offset, len)` (offsets relative to
/// `data`'s current logical view) in which `Struct` and `FixedSizeList` nodes carry zero offset.
fn normalize_window(data: &ArrayData, rel_offset: usize, len: usize) -> VortexResult<ArrayData> {
    match data.data_type() {
        DataType::Struct(_) => {
            // A struct's children are full arrays: logical parent row `i` corresponds to row
            // `data.offset() + rel_offset + i` of each child's own logical view.
            let child_start = data.offset() + rel_offset;
            let children = data
                .child_data()
                .iter()
                .map(|child| normalize_window(child, child_start, len))
                .collect::<VortexResult<Vec<_>>>()?;
            let nulls = data.nulls().map(|n| n.slice(rel_offset, len));
            ArrayData::builder(data.data_type().clone())
                .len(len)
                .nulls(nulls)
                .child_data(children)
                .build()
                .map_err(|e| vortex_err!("Failed to normalize struct ArrayData: {e}"))
        }
        DataType::FixedSizeList(_, size) => {
            let size = *size as usize;
            let child_start = (data.offset() + rel_offset) * size;
            let child = normalize_window(&data.child_data()[0], child_start, len * size)?;
            let nulls = data.nulls().map(|n| n.slice(rel_offset, len));
            ArrayData::builder(data.data_type().clone())
                .len(len)
                .nulls(nulls)
                .child_data(vec![child])
                .build()
                .map_err(|e| vortex_err!("Failed to normalize fixed-size list ArrayData: {e}"))
        }
        _ => {
            // For all other types `ArrayData::slice` is a plain offset bump, which downstream
            // `From<ArrayData>` impls interpret correctly. The children (e.g. list or dictionary
            // values) are not row-aligned with this node, so normalize them in place.
            let sliced = if rel_offset != 0 || len != data.len() {
                data.slice(rel_offset, len)
            } else {
                data.clone()
            };
            if !sliced.child_data().iter().any(needs_normalization) {
                return Ok(sliced);
            }
            let children = sliced
                .child_data()
                .iter()
                .map(|child| normalize_window(child, 0, child.len()))
                .collect::<VortexResult<Vec<_>>>()?;
            sliced
                .into_builder()
                .child_data(children)
                .build()
                .map_err(|e| vortex_err!("Failed to normalize nested ArrayData: {e}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::Array;
    use arrow_array::ArrayRef;
    use arrow_array::Int32Array;
    use arrow_array::cast::AsArray;
    use arrow_array::make_array;
    use arrow_array::types::Int32Type;
    use arrow_buffer::BooleanBuffer;
    use arrow_data::ArrayData;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Fields;

    use super::normalize_array_data;

    fn struct_dtype(fields: Fields) -> DataType {
        DataType::Struct(fields)
    }

    /// Build `ArrayData` the way the C data interface import does: full-length children with the
    /// parent offset *not* yet applied to them.
    fn ffi_style_struct(values: &[i32], offset: usize, len: usize) -> ArrayData {
        let child = Int32Array::from(values.to_vec()).into_data();
        let fields = Fields::from(vec![Field::new("c", DataType::Int32, false)]);
        // SAFETY: mirrors the unchecked construction performed by arrow's FFI import.
        unsafe {
            ArrayData::new_unchecked(
                struct_dtype(fields),
                len,
                Some(0),
                None,
                offset,
                vec![],
                vec![child],
            )
        }
    }

    #[test]
    fn normalize_sliced_struct_of_struct() {
        // {a: {c: i32}} with parent offset 1, mirroring a sliced struct<struct> from FFI.
        let inner = ffi_style_struct(&[1, 2, 3, 4], 0, 4);
        let fields = Fields::from(vec![Field::new("a", inner.data_type().clone(), false)]);
        let outer = unsafe {
            ArrayData::new_unchecked(
                struct_dtype(fields),
                2,
                Some(0),
                None,
                1,
                vec![],
                vec![inner],
            )
        };

        // Without normalization `make_array` panics with "end <= self.len()".
        let normalized = normalize_array_data(outer).unwrap();
        let array: ArrayRef = make_array(normalized);
        let outer_struct = array.as_struct();
        assert_eq!(outer_struct.len(), 2);
        let inner_struct = outer_struct.column(0).as_struct();
        let c = inner_struct.column(0).as_primitive::<Int32Type>();
        assert_eq!(c.values(), &[2, 3]);
    }

    #[test]
    fn normalize_sliced_fixed_size_list_of_struct() {
        // fixed_size_list<struct<c: i32>, 2> with offset 1: rows [[3, 4]] after the slice.
        let child = ffi_style_struct(&[1, 2, 3, 4], 0, 4);
        let fsl_dtype = DataType::FixedSizeList(
            Arc::new(Field::new("item", child.data_type().clone(), false)),
            2,
        );
        let fsl = unsafe {
            ArrayData::new_unchecked(fsl_dtype, 1, Some(0), None, 1, vec![], vec![child])
        };

        let normalized = normalize_array_data(fsl).unwrap();
        let array: ArrayRef = make_array(normalized);
        assert_eq!(array.len(), 1);
        let fsl_array = array.as_fixed_size_list();
        let values = fsl_array.values().as_struct();
        let c = values.column(0).as_primitive::<Int32Type>();
        assert_eq!(c.values(), &[3, 4]);
    }

    #[test]
    fn normalize_preserves_nulls() {
        let child = Int32Array::from(vec![1, 2, 3, 4]).into_data();
        let fields = Fields::from(vec![Field::new("c", DataType::Int32, false)]);
        let validity = BooleanBuffer::from(vec![true, false, true, true]);
        let data = unsafe {
            ArrayData::new_unchecked(
                struct_dtype(fields),
                2,
                None,
                Some(validity.into_inner()),
                1,
                vec![],
                vec![child],
            )
        };

        let normalized = normalize_array_data(data).unwrap();
        assert_eq!(normalized.offset(), 0);
        let array: ArrayRef = make_array(normalized);
        assert_eq!(array.len(), 2);
        assert!(array.is_null(0));
        assert!(array.is_valid(1));
    }

    /// Demonstrates the upstream arrow-rs inconsistency that makes normalization necessary.
    /// If this test starts failing after an arrow upgrade, the workaround can likely be removed.
    #[test]
    #[should_panic(expected = "end <= self.len()")]
    fn make_array_panics_without_normalization() {
        let inner = ffi_style_struct(&[1, 2, 3, 4], 0, 4);
        let fields = Fields::from(vec![Field::new("a", inner.data_type().clone(), false)]);
        let outer = unsafe {
            ArrayData::new_unchecked(
                struct_dtype(fields),
                2,
                Some(0),
                None,
                1,
                vec![],
                vec![inner],
            )
        };
        drop(make_array(outer));
    }

    #[test]
    fn normalize_no_op_for_plain_data() {
        let data = Int32Array::from(vec![1, 2, 3]).into_data();
        let normalized = normalize_array_data(data.clone()).unwrap();
        assert_eq!(normalized, data);
    }
}
