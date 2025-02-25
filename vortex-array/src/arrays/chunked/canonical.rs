use vortex_error::VortexResult;

use super::ChunkedArray;
use crate::builders::{builder_with_capacity, ArrayBuilder};
use crate::{Array as _, ArrayCanonicalImpl, Canonical};

impl ArrayCanonicalImpl for ChunkedArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        let mut builder = builder_with_capacity(self.dtype(), self.len());
        self.append_to_builder(builder.as_mut())?;
        builder.finish().to_canonical()
    }

    fn _append_to_builder(&self, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        for chunk in self.chunks() {
            chunk.append_to_builder(builder)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::DType;
    use vortex_dtype::DType::{List, Primitive};
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::PType::I32;

    use crate::accessor::ArrayAccessor;
    use crate::array::Array;
    use crate::arrays::chunked::canonical::pack_views;
    use crate::arrays::{ChunkedArray, ListArray, PrimitiveArray, StructArray, VarBinViewArray};
    use crate::compute::{scalar_at, slice};
    use crate::validity::Validity;
    use crate::variants::StructArrayTrait;
    use crate::ToCanonical;

    fn stringview_array() -> VarBinViewArray {
        VarBinViewArray::from_iter_str(["foo", "bar", "baz", "quak"])
    }

    #[test]
    pub fn pack_sliced_varbin() {
        let array1 = slice(&stringview_array(), 1, 3).unwrap();
        let array2 = slice(&stringview_array(), 2, 4).unwrap();
        let packed = pack_views(
            &[array1, array2],
            &DType::Utf8(NonNullable),
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
        let canonical_struct = chunked.to_struct().unwrap();
        let canonical_varbin = canonical_struct
            .maybe_null_field_by_idx(0)
            .unwrap()
            .to_varbinview()
            .unwrap();
        let original_varbin = struct_array
            .maybe_null_field_by_idx(0)
            .unwrap()
            .to_varbinview()
            .unwrap();
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

        assert_eq!(
            scalar_at(&l1, 0).unwrap(),
            scalar_at(&canon_values, 0).unwrap()
        );
        assert_eq!(
            scalar_at(&l2, 0).unwrap(),
            scalar_at(&canon_values, 1).unwrap()
        );
    }
}
