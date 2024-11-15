use vortex_dtype::field::Field;
use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexExpect, VortexResult};
use vortex_scalar::StructScalar;

use crate::array::sparse::SparseArray;
use crate::variants::{
    ArrayVariants, BinaryArrayTrait, BoolArrayTrait, ExtensionArrayTrait, ListArrayTrait,
    NullArrayTrait, PrimitiveArrayTrait, StructArrayTrait, Utf8ArrayTrait,
};
use crate::{ArrayDType, ArrayData, IntoArrayData};

/// Sparse arrays support all DTypes
impl ArrayVariants for SparseArray {
    fn as_null_array(&self) -> Option<&dyn NullArrayTrait> {
        matches!(self.dtype(), DType::Null).then_some(self)
    }

    fn as_bool_array(&self) -> Option<&dyn BoolArrayTrait> {
        matches!(self.dtype(), DType::Bool(_)).then_some(self)
    }

    fn as_primitive_array(&self) -> Option<&dyn PrimitiveArrayTrait> {
        matches!(self.dtype(), DType::Primitive(..)).then_some(self)
    }

    fn as_utf8_array(&self) -> Option<&dyn Utf8ArrayTrait> {
        matches!(self.dtype(), DType::Utf8(_)).then_some(self)
    }

    fn as_binary_array(&self) -> Option<&dyn BinaryArrayTrait> {
        matches!(self.dtype(), DType::Binary(_)).then_some(self)
    }

    fn as_struct_array(&self) -> Option<&dyn StructArrayTrait> {
        matches!(self.dtype(), DType::Struct(..)).then_some(self)
    }

    fn as_list_array(&self) -> Option<&dyn ListArrayTrait> {
        matches!(self.dtype(), DType::List(..)).then_some(self)
    }

    fn as_extension_array(&self) -> Option<&dyn ExtensionArrayTrait> {
        matches!(self.dtype(), DType::Extension(..)).then_some(self)
    }
}

impl NullArrayTrait for SparseArray {}

impl BoolArrayTrait for SparseArray {
    fn invert(&self) -> VortexResult<ArrayData> {
        let inverted_fill = self.fill_value().as_bool()?.map(|v| !v);
        SparseArray::try_new(
            self.indices(),
            self.values().with_dyn(|a| {
                a.as_bool_array()
                    .ok_or_else(|| vortex_err!("Not a bool array"))
                    .and_then(|b| b.invert())
            })?,
            self.len(),
            inverted_fill.into(),
        )
        .map(|a| a.into_array())
    }

    fn maybe_null_indices_iter(&self) -> Box<dyn Iterator<Item = usize>> {
        // TODO(robert): Indices of the array can include true and false values, fill value can be true
        todo!()
    }

    fn maybe_null_slices_iter(&self) -> Box<dyn Iterator<Item = (usize, usize)>> {
        todo!()
    }
}

impl PrimitiveArrayTrait for SparseArray {}

impl Utf8ArrayTrait for SparseArray {}

impl BinaryArrayTrait for SparseArray {}

impl StructArrayTrait for SparseArray {
    fn field(&self, idx: usize) -> Option<ArrayData> {
        let values = self
            .values()
            .with_dyn(|s| s.as_struct_array().and_then(|s| s.field(idx)))?;
        let scalar = StructScalar::try_new(self.dtype(), self.fill_value())
            .ok()?
            .field_by_idx(idx)?;

        Some(
            SparseArray::try_new_with_offset(
                self.indices().clone(),
                values,
                self.len(),
                self.indices_offset(),
                scalar.value().clone(),
            )
            .ok()?
            .into_array(),
        )
    }

    fn project(&self, projection: &[Field]) -> VortexResult<ArrayData> {
        let values = self.values().with_dyn(|s| {
            s.as_struct_array()
                .ok_or_else(|| vortex_err!("Chunk was not a StructArray"))?
                .project(projection)
        })?;
        let scalar = StructScalar::try_new(self.dtype(), self.fill_value())?.project(projection)?;

        SparseArray::try_new_with_offset(
            self.indices().clone(),
            values,
            self.len(),
            self.indices_offset(),
            scalar.value().clone(),
        )
        .map(|a| a.into_array())
    }
}

impl ListArrayTrait for SparseArray {}

impl ExtensionArrayTrait for SparseArray {
    fn storage_data(&self) -> ArrayData {
        SparseArray::try_new_with_offset(
            self.indices().clone(),
            self.values()
                .with_dyn(|a| a.as_extension_array_unchecked().storage_data()),
            self.len(),
            self.indices_offset(),
            self.fill_value().clone(),
        )
        .vortex_expect("Failed to create new sparse array")
        .into_array()
    }
}

#[cfg(test)]
mod tests {
    use crate::array::{BoolArray, PrimitiveArray, SparseArray};
    use crate::{IntoArrayData, IntoArrayVariant};

    #[test]
    fn invert_bools_non_null_fill() {
        let sparse_bools = SparseArray::try_new(
            PrimitiveArray::from(vec![0u64]).into_array(),
            BoolArray::from(vec![false]).into_array(),
            2,
            true.into(),
        )
        .unwrap()
        .into_array();
        let inverted = sparse_bools
            .with_dyn(|a| a.as_bool_array_unchecked().invert())
            .unwrap();
        assert_eq!(
            inverted
                .into_bool()
                .unwrap()
                .boolean_buffer()
                .iter()
                .collect::<Vec<_>>(),
            vec![true, false]
        );
    }
}
