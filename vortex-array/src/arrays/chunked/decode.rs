// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferMut;
use vortex_dtype::{DType, Nullability, PType, StructFields};
use vortex_error::{VortexExpect, VortexUnwrap, vortex_err};

use super::ChunkedArray;
use crate::arrays::{ChunkedVTable, ListArray, PrimitiveArray, StructArray};
use crate::builders::{ArrayBuilder, builder_with_capacity};
use crate::compute::cast;
use crate::validity::Validity;
use crate::vtable::CanonicalVTable;
use crate::{Array as _, ArrayRef, Canonical, IntoArray, ToCanonical};

impl CanonicalVTable<ChunkedVTable> for ChunkedVTable {
    fn canonicalize(array: &ChunkedArray) -> Canonical {
        if array.nchunks() == 0 {
            return Canonical::empty(array.dtype());
        }
        if array.nchunks() == 1 {
            return array.chunks()[0].to_canonical();
        }

        match array.dtype() {
            DType::Struct(struct_dtype, _) => {
                let struct_array = swizzle_struct_chunks(
                    array.chunks(),
                    Validity::copy_from_array(array.as_ref()),
                    struct_dtype,
                );
                Canonical::Struct(struct_array)
            }
            DType::List(elem_dtype, _) => Canonical::List(pack_lists(
                array.chunks(),
                Validity::copy_from_array(array.as_ref()),
                elem_dtype,
            )),
            _ => {
                let mut builder = builder_with_capacity(array.dtype(), array.len());
                array.append_to_builder(builder.as_mut());
                builder.finish_into_canonical()
            }
        }
    }

    fn append_to_builder(array: &ChunkedArray, builder: &mut dyn ArrayBuilder) {
        for chunk in array.chunks() {
            chunk.append_to_builder(builder);
        }
    }
}

/// Swizzle the pointers within a ChunkedArray of StructArrays to instead be a single
/// StructArray, where the Array for each Field is a ChunkedArray.
fn swizzle_struct_chunks(
    chunks: &[ArrayRef],
    validity: Validity,
    struct_dtype: &StructFields,
) -> StructArray {
    let len = chunks.iter().map(|chunk| chunk.len()).sum();
    let mut field_arrays = Vec::new();

    for (field_idx, field_dtype) in struct_dtype.fields().enumerate() {
        let field_chunks = chunks
            .iter()
            .map(|c| {
                c.to_struct()
                    .fields()
                    .get(field_idx)
                    .vortex_expect("Invalid field index")
                    .to_array()
            })
            .collect::<Vec<_>>();
        let field_array = unsafe { ChunkedArray::new_unchecked(field_chunks, field_dtype.clone()) };
        field_arrays.push(field_array.into_array());
    }

    StructArray::new_unchecked(field_arrays, struct_dtype.clone(), len, validity)
}

fn pack_lists(chunks: &[ArrayRef], validity: Validity, elem_dtype: &DType) -> ListArray {
    let len: usize = chunks.iter().map(|c| c.len()).sum();
    let mut offsets = BufferMut::<u64>::with_capacity(len + 1);
    offsets.push(0);
    let mut elements = Vec::new();

    for chunk in chunks {
        let chunk = chunk.to_list();
        // TODO: handle i32 offsets if they fit.
        let offsets_arr = cast(
            chunk.offsets(),
            &DType::Primitive(PType::U64, Nullability::NonNullable),
        )
        .vortex_expect("Must fit array offsets in u64")
        .to_primitive();

        let first_offset_value: usize =
            usize::try_from(&offsets_arr.scalar_at(0)).vortex_expect("Offset must be a usize");
        let last_offset_value: usize =
            usize::try_from(&offsets_arr.scalar_at(offsets_arr.len() - 1))
                .vortex_expect("Offset must be a usize");
        elements.push(
            chunk
                .elements()
                .slice(first_offset_value..last_offset_value),
        );

        let adjustment_from_previous = *offsets
            .last()
            .ok_or_else(|| vortex_err!("List offsets must have at least one element"))
            .vortex_unwrap();
        offsets.extend_trusted(
            offsets_arr
                .as_slice::<u64>()
                .iter()
                .skip(1)
                .map(|off| off + adjustment_from_previous - first_offset_value as u64),
        );
    }
    let chunked_elements =
        unsafe { ChunkedArray::new_unchecked(elements, elem_dtype.clone()) }.into_array();
    let offsets = PrimitiveArray::new(offsets.freeze(), Validity::NonNullable);

    unsafe { ListArray::new_unchecked(chunked_elements, offsets.into_array(), validity) }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::DType::{List, Primitive};
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::PType::I32;

    use crate::accessor::ArrayAccessor;
    use crate::arrays::{ChunkedArray, ListArray, PrimitiveArray, StructArray, VarBinViewArray};
    use crate::validity::Validity;
    use crate::{IntoArray, ToCanonical};

    #[test]
    pub fn pack_nested_structs() {
        let struct_array = StructArray::try_new(
            vec!["a".into()].into(),
            vec![VarBinViewArray::from_iter_str(["foo", "bar", "baz", "quak"]).into_array()],
            4,
            Validity::NonNullable,
        )
        .unwrap();
        let dtype = struct_array.dtype().clone();
        let chunked = ChunkedArray::try_new(
            vec![
                ChunkedArray::try_new(vec![struct_array.to_array()], dtype.clone())
                    .unwrap()
                    .into_array(),
            ],
            dtype,
        )
        .unwrap()
        .into_array();
        let canonical_struct = chunked.to_struct();
        let canonical_varbin = canonical_struct.fields()[0].to_varbinview();
        let original_varbin = struct_array.fields()[0].to_varbinview();
        let orig_values = original_varbin
            .with_iterator(|it| it.map(|a| a.map(|v| v.to_vec())).collect::<Vec<_>>())
            .unwrap();
        let canon_values = canonical_varbin
            .with_iterator(|it| it.map(|a| a.map(|v| v.to_vec())).collect::<Vec<_>>())
            .unwrap();
        assert_eq!(orig_values, canon_values);
    }

    #[test]
    pub fn pack_nested_lists() {
        let l1 = ListArray::try_new(
            PrimitiveArray::from_iter([1, 2, 3, 4]).into_array(),
            PrimitiveArray::from_iter([0, 3]).into_array(),
            Validity::NonNullable,
        )
        .unwrap();

        let l2 = ListArray::try_new(
            PrimitiveArray::from_iter([5, 6]).into_array(),
            PrimitiveArray::from_iter([0, 2]).into_array(),
            Validity::NonNullable,
        )
        .unwrap();

        let chunked_list = ChunkedArray::try_new(
            vec![l1.clone().into_array(), l2.clone().into_array()],
            List(Arc::new(Primitive(I32, NonNullable)), NonNullable),
        );

        let canon_values = chunked_list.unwrap().to_list();

        assert_eq!(l1.scalar_at(0), canon_values.scalar_at(0));
        assert_eq!(l2.scalar_at(0), canon_values.scalar_at(1));
    }
}
