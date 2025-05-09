use std::sync::Arc;

use vortex_buffer::BufferMut;
use vortex_dtype::{DType, Nullability, PType, StructDType};
use vortex_error::{VortexExpect, VortexResult, vortex_err};

use super::ChunkedArray;
use crate::arrays::{ListArray, PrimitiveArray, StructArray};
use crate::builders::{ArrayBuilder, builder_with_capacity};
use crate::compute::cast;
use crate::validity::Validity;
use crate::{Array as _, ArrayCanonicalImpl, ArrayRef, Canonical, ToCanonical};

impl ArrayCanonicalImpl for ChunkedArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        if self.nchunks() == 0 {
            return Ok(Canonical::empty(self.dtype()));
        }
        if self.nchunks() == 1 {
            return self.chunks()[0].to_canonical();
        }
        match self.dtype() {
            DType::Struct(struct_dtype, _) => {
                let struct_array = swizzle_struct_chunks(
                    self.chunks(),
                    Validity::copy_from_array(self)?,
                    struct_dtype,
                )?;
                Ok(Canonical::Struct(struct_array))
            }
            DType::List(elem, _) => Ok(Canonical::List(pack_lists(
                self.chunks(),
                Validity::copy_from_array(self)?,
                elem,
            )?)),
            _ => {
                let mut builder = builder_with_capacity(self.dtype(), self.len());
                self.append_to_builder(builder.as_mut())?;
                builder.finish().to_canonical()
            }
        }
    }

    fn _append_to_builder(&self, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        for chunk in self.chunks() {
            chunk.append_to_builder(builder)?;
        }
        Ok(())
    }
}

/// Swizzle the pointers within a ChunkedArray of StructArrays to instead be a single
/// StructArray, where the Array for each Field is a ChunkedArray.
fn swizzle_struct_chunks(
    chunks: &[ArrayRef],
    validity: Validity,
    struct_dtype: &Arc<StructDType>,
) -> VortexResult<StructArray> {
    let len = chunks.iter().map(|chunk| chunk.len()).sum();
    let mut field_arrays = Vec::new();

    for (field_idx, field_dtype) in struct_dtype.fields().enumerate() {
        let field_chunks = chunks
            .iter()
            .map(|c| {
                c.to_struct()
                    .vortex_expect("Chunk was not a StructArray")
                    .fields()
                    .get(field_idx)
                    .vortex_expect("Invalid field index")
                    .to_array()
            })
            .collect::<Vec<_>>();
        let field_array = ChunkedArray::try_new(field_chunks, field_dtype.clone())?;
        field_arrays.push(field_array.into_array());
    }

    StructArray::try_new_with_dtype(field_arrays, struct_dtype.clone(), len, validity)
}

fn pack_lists(
    chunks: &[ArrayRef],
    validity: Validity,
    elem_dtype: &DType,
) -> VortexResult<ListArray> {
    let len: usize = chunks.iter().map(|c| c.len()).sum();
    let mut offsets = BufferMut::<i64>::with_capacity(len + 1);
    offsets.push(0);
    let mut elements = Vec::new();

    for chunk in chunks {
        let chunk = chunk.to_list()?;
        // TODO: handle i32 offsets if they fit.
        let offsets_arr = cast(
            chunk.offsets(),
            &DType::Primitive(PType::I64, Nullability::NonNullable),
        )?
        .to_primitive()?;

        let first_offset_value: usize = usize::try_from(&offsets_arr.scalar_at(0)?)?;
        let last_offset_value: usize =
            usize::try_from(&offsets_arr.scalar_at(offsets_arr.len() - 1)?)?;
        elements.push(
            chunk
                .elements()
                .slice(first_offset_value, last_offset_value)?,
        );

        let adjustment_from_previous = *offsets
            .last()
            .ok_or_else(|| vortex_err!("List offsets must have at least one element"))?;
        offsets.extend(
            offsets_arr
                .as_slice::<i64>()
                .iter()
                .skip(1)
                .map(|off| off + adjustment_from_previous - first_offset_value as i64),
        );
    }
    let chunked_elements = ChunkedArray::try_new(elements, elem_dtype.clone())?.into_array();
    let offsets = PrimitiveArray::new(offsets.freeze(), Validity::NonNullable);

    ListArray::try_new(chunked_elements, offsets.into_array(), validity)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::DType::{List, Primitive};
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::PType::I32;

    use crate::ToCanonical;
    use crate::accessor::ArrayAccessor;
    use crate::array::Array;
    use crate::arrays::{ChunkedArray, ListArray, PrimitiveArray, StructArray, VarBinViewArray};
    use crate::validity::Validity;

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
        let canonical_struct = chunked.to_struct().unwrap();
        let canonical_varbin = canonical_struct.fields()[0].to_varbinview().unwrap();
        let original_varbin = struct_array.fields()[0].to_varbinview().unwrap();
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

        let canon_values = chunked_list.unwrap().to_list().unwrap();

        assert_eq!(l1.scalar_at(0).unwrap(), canon_values.scalar_at(0).unwrap());
        assert_eq!(l2.scalar_at(0).unwrap(), canon_values.scalar_at(1).unwrap());
    }
}
