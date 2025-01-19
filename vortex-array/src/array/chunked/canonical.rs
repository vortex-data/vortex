use arrow_buffer::BooleanBufferBuilder;
use vortex_buffer::BufferMut;
use vortex_dtype::{match_each_native_ptype, DType, NativePType, Nullability, PType, StructDType};
use vortex_error::{vortex_panic, VortexExpect, VortexUnwrap};

use crate::array::chunked::ChunkedArray;
use crate::array::extension::ExtensionArray;
use crate::array::null::NullArray;
use crate::array::primitive::PrimitiveArray;
use crate::array::struct_::StructArray;
use crate::array::{BinaryView, BoolArray, ListArray, VarBinViewArray};
use crate::compute::{scalar_at, slice, try_cast};
use crate::validity::Validity;
use crate::{
    ArrayDType, ArrayData, ArrayLen, ArrayValidity, Canonical, IntoArrayData, IntoArrayVariant,
    IntoCanonical,
};

impl IntoCanonical for ChunkedArray {
    fn into_canonical(self) -> Canonical {
        let validity = self
            .logical_validity()
            .into_validity(self.dtype().nullability());
        try_canonicalize_chunks(self.chunks().collect(), validity, self.dtype())
    }
}

pub(crate) fn try_canonicalize_chunks(
    chunks: Vec<ArrayData>,
    validity: Validity,
    dtype: &DType,
) -> Canonical {
    let mismatched = chunks
        .iter()
        .filter(|chunk| !chunk.dtype().eq(dtype))
        .collect::<Vec<_>>();
    if !mismatched.is_empty() {
        vortex_panic!(
            "Expected {} dtype but got {:?} that don't match",
            dtype,
            mismatched
        )
    }

    match dtype {
        // Structs can have their internal field pointers swizzled to push the chunking down
        // one level internally without copying or decompressing any data.
        DType::Struct(struct_dtype, _) => {
            let struct_array = swizzle_struct_chunks(chunks.as_slice(), validity, struct_dtype);
            Canonical::Struct(struct_array)
        }

        // Extension arrays are containers that wraps an inner storage array with some metadata.
        // We delegate to the canonical format of the internal storage array for every chunk, and
        // push the chunking down into the inner storage array.
        //
        //  Input:
        //  ------
        //
        //                  ChunkedArray
        //                 /            \
        //                /              \
        //         ExtensionArray     ExtensionArray
        //             |                   |
        //          storage             storage
        //
        //
        //  Output:
        //  ------
        //
        //                 ExtensionArray
        //                      |
        //                 ChunkedArray
        //                /             \
        //          storage             storage
        //
        DType::Extension(ext_dtype) => {
            // Recursively apply canonicalization and packing to the storage array backing
            // each chunk of the extension array.
            let storage_chunks: Vec<ArrayData> = chunks
                .iter()
                // Extension-typed arrays can be compressed into something that is not an
                // ExtensionArray, so we should canonicalize each chunk into ExtensionArray first.
                .map(|chunk| chunk.clone().into_canonical_extension().storage())
                .collect::<Vec<_>>();
            let storage_dtype = ext_dtype.storage_dtype().clone();
            let chunked_storage = ChunkedArray::try_new(storage_chunks, storage_dtype)
                .vortex_unwrap()
                .into_array();

            Canonical::Extension(ExtensionArray::new(ext_dtype.clone(), chunked_storage))
        }

        DType::List(..) => {
            // TODO(joe): improve performance, use a listview, once it exists

            let list = pack_lists(chunks.as_slice(), validity, dtype);
            Canonical::List(list)
        }

        DType::Bool(_) => {
            let bool_array = pack_bools(chunks.as_slice(), validity);
            Canonical::Bool(bool_array)
        }
        DType::Primitive(ptype, _) => {
            match_each_native_ptype!(ptype, |$P| {
                let prim_array = pack_primitives::<$P>(chunks.as_slice(), validity);
                Canonical::Primitive(prim_array)
            })
        }
        DType::Utf8(_) => {
            let varbin_array = pack_views(chunks.as_slice(), dtype, validity);
            Canonical::VarBinView(varbin_array)
        }
        DType::Binary(_) => {
            let varbin_array = pack_views(chunks.as_slice(), dtype, validity);
            Canonical::VarBinView(varbin_array)
        }
        DType::Null => {
            let len = chunks.iter().map(|chunk| chunk.len()).sum();
            Canonical::Null(NullArray::new(len))
        }
    }
}

fn pack_lists(chunks: &[ArrayData], validity: Validity, dtype: &DType) -> ListArray {
    let len: usize = chunks.iter().map(|c| c.len()).sum();
    let mut offsets = BufferMut::<i64>::with_capacity(len + 1);
    offsets.push(0);
    let mut elements = Vec::new();
    let elem_dtype = dtype
        .as_list_element()
        .vortex_expect("ListArray must have List dtype");

    for chunk in chunks {
        let chunk = chunk.clone().into_canonical_list();
        // TODO: handle i32 offsets if they fit.
        let offsets_arr = try_cast(
            chunk.offsets(),
            &DType::Primitive(PType::I64, Nullability::NonNullable),
        )
        .vortex_unwrap()
        .into_canonical_primitive();

        let first_offset_value: usize =
            usize::try_from(&scalar_at(offsets_arr.as_ref(), 0).vortex_unwrap()).vortex_unwrap();
        let last_offset_value: usize = usize::try_from(
            &scalar_at(offsets_arr.as_ref(), offsets_arr.len() - 1).vortex_unwrap(),
        )
        .vortex_unwrap();
        elements
            .push(slice(chunk.elements(), first_offset_value, last_offset_value).vortex_unwrap());

        let adjustment_from_previous = *offsets.last().vortex_expect("Offsets must not be empty");
        offsets.extend(
            offsets_arr
                .as_slice::<i64>()
                .iter()
                .skip(1)
                .map(|off| off + adjustment_from_previous - first_offset_value as i64),
        );
    }
    let chunked_elements = ChunkedArray::try_new(elements, elem_dtype.clone())
        .vortex_unwrap()
        .into_array();
    let offsets = PrimitiveArray::new(offsets.freeze(), Validity::NonNullable);

    ListArray::try_new(chunked_elements, offsets.into_array(), validity).vortex_unwrap()
}

/// Swizzle the pointers within a ChunkedArray of StructArrays to instead be a single
/// StructArray, where the Array for each Field is a ChunkedArray.
///
/// It is expected this function is only called from [try_canonicalize_chunks], and thus all chunks have
/// been checked to have the same DType already.
fn swizzle_struct_chunks(
    chunks: &[ArrayData],
    validity: Validity,
    struct_dtype: &StructDType,
) -> StructArray {
    let len = chunks.iter().map(|chunk| chunk.len()).sum();
    let mut field_arrays = Vec::new();

    for (field_idx, field_dtype) in struct_dtype.dtypes().enumerate() {
        let field_chunks = chunks
            .iter()
            .map(|c| {
                c.as_struct_array()
                    .vortex_expect("Chunk was not a StructArray")
                    .maybe_null_field_by_idx(field_idx)
                    .vortex_expect("Accessing struct child by dtype")
            })
            .collect::<Vec<_>>();
        let field_array = ChunkedArray::try_new(field_chunks, field_dtype.clone())
            .vortex_expect("Chunked array canonicalizing");
        field_arrays.push(field_array.into_array());
    }

    StructArray::try_new(struct_dtype.names().clone(), field_arrays, len, validity)
        .vortex_expect("Canonicalizing")
}

/// Builds a new [BoolArray] by repacking the values from the chunks in a single contiguous array.
///
/// It is expected this function is only called from [try_canonicalize_chunks], and thus all chunks have
/// been checked to have the same DType already.
fn pack_bools(chunks: &[ArrayData], validity: Validity) -> BoolArray {
    let len = chunks.iter().map(|chunk| chunk.len()).sum();
    let mut buffer = BooleanBufferBuilder::new(len);
    for chunk in chunks {
        let chunk = chunk.clone().into_canonical_bool();
        buffer.append_buffer(&chunk.boolean_buffer());
    }

    BoolArray::try_new(buffer.finish(), validity).vortex_unwrap()
}

/// Builds a new [PrimitiveArray] by repacking the values from the chunks into a single
/// contiguous array.
///
/// It is expected this function is only called from [try_canonicalize_chunks], and thus all chunks have
/// been checked to have the same DType already.
fn pack_primitives<T: NativePType>(chunks: &[ArrayData], validity: Validity) -> PrimitiveArray {
    let total_len = chunks.iter().map(|a| a.len()).sum();
    let mut buffer = BufferMut::with_capacity(total_len);
    for chunk in chunks {
        let chunk = chunk.clone().into_canonical_primitive();
        buffer.extend_from_slice(chunk.as_slice::<T>());
    }
    PrimitiveArray::new(buffer.freeze(), validity)
}

/// Builds a new [VarBinViewArray] by repacking the values from the chunks into a single
/// contiguous array.
///
/// It is expected this function is only called from [try_canonicalize_chunks], and thus all chunks have
/// been checked to have the same DType already.
fn pack_views(chunks: &[ArrayData], dtype: &DType, validity: Validity) -> VarBinViewArray {
    let total_len = chunks.iter().map(|a| a.len()).sum();
    let mut views = BufferMut::with_capacity(total_len);
    let mut buffers = Vec::new();
    for chunk in chunks {
        // Each chunk's views have buffer IDs that are zero-referenced.
        // As part of the packing operation, we need to rewrite them to be referenced to the global
        // merged buffers list.
        let buffers_offset = u32::try_from(buffers.len()).vortex_unwrap();
        let canonical_chunk = chunk.clone().into_canonical_varbinview();
        buffers.extend(canonical_chunk.buffers());

        for view in canonical_chunk.views().iter() {
            if view.is_inlined() {
                // Inlined views can be copied directly into the output
                views.push(*view);
            } else {
                // Referencing views must have their buffer_index adjusted with new offsets
                let view_ref = view.as_view();
                views.push(BinaryView::new_view(
                    view.len(),
                    *view_ref.prefix(),
                    buffers_offset + view_ref.buffer_index(),
                    view_ref.offset(),
                ));
            }
        }
    }

    VarBinViewArray::try_new(views.freeze(), buffers, dtype.clone(), validity).vortex_unwrap()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::DType;
    use vortex_dtype::DType::{List, Primitive};
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::PType::I32;

    use crate::accessor::ArrayAccessor;
    use crate::array::chunked::canonical::pack_views;
    use crate::array::{ChunkedArray, ListArray, PrimitiveArray, StructArray, VarBinViewArray};
    use crate::compute::{scalar_at, slice};
    use crate::validity::Validity;
    use crate::variants::StructArrayTrait;
    use crate::{ArrayDType, ArrayLen, IntoArrayData, IntoArrayVariant, ToArrayData};

    fn stringview_array() -> VarBinViewArray {
        VarBinViewArray::from_iter_str(["foo", "bar", "baz", "quak"])
    }

    #[test]
    pub fn pack_sliced_varbin() {
        let array1 = slice(stringview_array().as_ref(), 1, 3).unwrap();
        let array2 = slice(stringview_array().as_ref(), 2, 4).unwrap();
        let packed = pack_views(
            &[array1, array2],
            &DType::Utf8(NonNullable),
            Validity::NonNullable,
        );
        assert_eq!(packed.len(), 4);
        let values = packed.with_iterator(|iter| {
            iter.flatten()
                .map(|v| unsafe { String::from_utf8_unchecked(v.to_vec()) })
                .collect::<Vec<_>>()
        });
        assert_eq!(values, &["bar", "baz", "baz", "quak"]);
    }

    #[test]
    pub fn pack_nested_structs() {
        let struct_array = StructArray::try_new(
            vec!["a".into()].into(),
            vec![stringview_array().into_array()],
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
        let canonical_struct = chunked.into_canonical_struct();
        let canonical_varbin = canonical_struct
            .maybe_null_field_by_idx(0)
            .unwrap()
            .into_canonical_varbinview();
        let original_varbin = struct_array
            .maybe_null_field_by_idx(0)
            .unwrap()
            .into_canonical_varbinview();
        let orig_values = original_varbin
            .with_iterator(|it| it.map(|a| a.map(|v| v.to_vec())).collect::<Vec<_>>());
        let canon_values = canonical_varbin
            .with_iterator(|it| it.map(|a| a.map(|v| v.to_vec())).collect::<Vec<_>>());
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

        let canon_values = chunked_list.unwrap().into_canonical_list();

        assert_eq!(
            scalar_at(l1, 0).unwrap(),
            scalar_at(canon_values.clone(), 0).unwrap()
        );
        assert_eq!(
            scalar_at(l2, 0).unwrap(),
            scalar_at(canon_values, 1).unwrap()
        );
    }
}
