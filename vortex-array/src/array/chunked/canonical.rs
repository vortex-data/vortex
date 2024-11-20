use arrow_buffer::{BooleanBufferBuilder, Buffer, MutableBuffer, ScalarBuffer};
use vortex_dtype::{DType, PType, StructDType};
use vortex_error::{vortex_bail, vortex_err, ErrString, VortexResult};

use crate::array::chunked::ChunkedArray;
use crate::array::extension::ExtensionArray;
use crate::array::null::NullArray;
use crate::array::primitive::PrimitiveArray;
use crate::array::struct_::StructArray;
use crate::array::{BinaryView, BoolArray, VarBinViewArray};
use crate::validity::Validity;
use crate::{
    ArrayDType, ArrayData, ArrayValidity, Canonical, IntoArrayData, IntoArrayVariant, IntoCanonical,
};

impl IntoCanonical for ChunkedArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        let validity = if self.dtype().is_nullable() {
            self.logical_validity().into_validity()
        } else {
            Validity::NonNullable
        };
        try_canonicalize_chunks(self.chunks().collect(), validity, self.dtype())
    }
}

pub(crate) fn try_canonicalize_chunks(
    chunks: Vec<ArrayData>,
    validity: Validity,
    dtype: &DType,
) -> VortexResult<Canonical> {
    let mismatched = chunks
        .iter()
        .filter(|chunk| !chunk.dtype().eq(dtype))
        .collect::<Vec<_>>();
    if !mismatched.is_empty() {
        vortex_bail!(MismatchedTypes: dtype.clone(), ErrString::from(format!("{:?}", mismatched)))
    }

    match dtype {
        // Structs can have their internal field pointers swizzled to push the chunking down
        // one level internally without copying or decompressing any data.
        DType::Struct(struct_dtype, _) => {
            let struct_array = swizzle_struct_chunks(chunks.as_slice(), validity, struct_dtype)?;
            Ok(Canonical::Struct(struct_array))
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
                .map(|chunk| chunk.clone().into_extension().map(|ext| ext.storage()))
                .collect::<VortexResult<Vec<ArrayData>>>()?;
            let storage_dtype = ext_dtype.storage_dtype().clone();
            let chunked_storage =
                ChunkedArray::try_new(storage_chunks, storage_dtype)?.into_array();

            Ok(Canonical::Extension(ExtensionArray::new(
                ext_dtype.clone(),
                chunked_storage,
            )))
        }

        // TODO(aduffy): better list support
        DType::List(..) => {
            todo!()
        }

        DType::Bool(_) => {
            let bool_array = pack_bools(chunks.as_slice(), validity)?;
            Ok(Canonical::Bool(bool_array))
        }
        DType::Primitive(ptype, _) => {
            let prim_array = pack_primitives(chunks.as_slice(), *ptype, validity)?;
            Ok(Canonical::Primitive(prim_array))
        }
        DType::Utf8(_) => {
            let varbin_array = pack_views(chunks.as_slice(), dtype, validity)?;
            Ok(Canonical::VarBinView(varbin_array))
        }
        DType::Binary(_) => {
            let varbin_array = pack_views(chunks.as_slice(), dtype, validity)?;
            Ok(Canonical::VarBinView(varbin_array))
        }
        DType::Null => {
            let len = chunks.iter().map(|chunk| chunk.len()).sum();
            let null_array = NullArray::new(len);
            Ok(Canonical::Null(null_array))
        }
    }
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
) -> VortexResult<StructArray> {
    let len = chunks.iter().map(|chunk| chunk.len()).sum();
    let mut field_arrays = Vec::new();

    for (field_idx, field_dtype) in struct_dtype.dtypes().iter().enumerate() {
        let field_chunks = chunks.iter().map(|c| c.with_dyn(|d|
            d.as_struct_array_unchecked()
                .field(field_idx)
                .ok_or_else(|| vortex_err!("All chunks must have same dtype; missing field at index {}, current chunk dtype: {}", field_idx, c.dtype())),
        )).collect::<VortexResult<Vec<_>>>()?;
        let field_array = ChunkedArray::try_new(field_chunks, field_dtype.clone())?;
        field_arrays.push(field_array.into_array());
    }

    StructArray::try_new(struct_dtype.names().clone(), field_arrays, len, validity)
}

/// Builds a new [BoolArray] by repacking the values from the chunks in a single contiguous array.
///
/// It is expected this function is only called from [try_canonicalize_chunks], and thus all chunks have
/// been checked to have the same DType already.
fn pack_bools(chunks: &[ArrayData], validity: Validity) -> VortexResult<BoolArray> {
    let len = chunks.iter().map(|chunk| chunk.len()).sum();
    let mut buffer = BooleanBufferBuilder::new(len);
    for chunk in chunks {
        let chunk = chunk.clone().into_bool()?;
        buffer.append_buffer(&chunk.boolean_buffer());
    }

    BoolArray::try_new(buffer.finish(), validity)
}

/// Builds a new [PrimitiveArray] by repacking the values from the chunks into a single
/// contiguous array.
///
/// It is expected this function is only called from [try_canonicalize_chunks], and thus all chunks have
/// been checked to have the same DType already.
fn pack_primitives(
    chunks: &[ArrayData],
    ptype: PType,
    validity: Validity,
) -> VortexResult<PrimitiveArray> {
    let len: usize = chunks.iter().map(|chunk| chunk.len()).sum();
    let mut buffer = MutableBuffer::with_capacity(len * ptype.byte_width());
    for chunk in chunks {
        let chunk = chunk.clone().into_primitive()?;
        buffer.extend_from_slice(chunk.buffer());
    }

    Ok(PrimitiveArray::new(
        Buffer::from(buffer).into(),
        ptype,
        validity,
    ))
}

/// Builds a new [VarBinViewArray] by repacking the values from the chunks into a single
/// contiguous array.
///
/// It is expected this function is only called from [try_canonicalize_chunks], and thus all chunks have
/// been checked to have the same DType already.
fn pack_views(
    chunks: &[ArrayData],
    dtype: &DType,
    validity: Validity,
) -> VortexResult<VarBinViewArray> {
    let total_len = chunks.iter().map(|a| a.len()).sum();
    let mut views: Vec<u128> = Vec::with_capacity(total_len);
    let mut buffers = Vec::new();
    for chunk in chunks {
        // Each chunk's views have buffer IDs that are zero-referenced.
        // As part of the packing operation, we need to rewrite them to be referenced to the global
        // merged buffers list.
        let buffers_offset = buffers.len();
        let canonical_chunk = chunk.clone().into_varbinview()?;

        for buffer in canonical_chunk.buffers() {
            let canonical_buffer = buffer.into_canonical()?.into_primitive()?.into_array();
            buffers.push(canonical_buffer);
        }

        for view in canonical_chunk.binary_views()? {
            if view.is_inlined() {
                // Inlined views can be copied directly into the output
                views.push(view.as_u128());
            } else {
                // Referencing views must have their buffer_index adjusted with new offsets
                let view_ref = view.as_view();
                views.push(
                    BinaryView::new_view(
                        view.len(),
                        *view_ref.prefix(),
                        (buffers_offset as u32) + view_ref.buffer_index(),
                        view_ref.offset(),
                    )
                    .as_u128(),
                );
            }
        }
    }

    let views_buffer: Buffer = ScalarBuffer::<u128>::from(views).into_inner();
    VarBinViewArray::try_new(
        ArrayData::from(views_buffer),
        buffers,
        dtype.clone(),
        validity,
    )
}

#[cfg(test)]
mod tests {
    use vortex_dtype::{DType, Nullability};

    use crate::accessor::ArrayAccessor;
    use crate::array::chunked::canonical::pack_views;
    use crate::array::{ChunkedArray, StructArray, VarBinViewArray};
    use crate::compute::slice;
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
            &DType::Utf8(Nullability::NonNullable),
            Validity::NonNullable,
        )
        .unwrap();
        assert_eq!(packed.len(), 4);
        let values = packed
            .with_iterator(|iter| {
                iter.flatten()
                    .map(|v| unsafe { String::from_utf8_unchecked(v.to_vec()) })
                    .collect::<Vec<_>>()
            })
            .unwrap();
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
        let canonical_struct = chunked.into_struct().unwrap();
        let canonical_varbin = canonical_struct
            .field(0)
            .unwrap()
            .into_varbinview()
            .unwrap();
        let original_varbin = struct_array.field(0).unwrap().into_varbinview().unwrap();
        let orig_values = original_varbin
            .with_iterator(|it| it.map(|a| a.map(|v| v.to_vec())).collect::<Vec<_>>())
            .unwrap();
        let canon_values = canonical_varbin
            .with_iterator(|it| it.map(|a| a.map(|v| v.to_vec())).collect::<Vec<_>>())
            .unwrap();
        assert_eq!(orig_values, canon_values);
    }
}
