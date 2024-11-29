use vortex_dtype::field::Field;
use vortex_error::{vortex_err, VortexExpect, VortexResult};
use vortex_scalar::StructScalar;

use crate::array::sparse::SparseArray;
use crate::array::SparseEncoding;
use crate::variants::{
    BinaryArrayTrait, BoolArrayTrait, ExtensionArrayTrait, ListArrayTrait, NullArrayTrait,
    PrimitiveArrayTrait, StructArrayTrait, Utf8ArrayTrait, VariantsVTable,
};
use crate::{ArrayData, ArrayLen, IntoArrayData};

/// Sparse arrays support all DTypes
impl VariantsVTable<SparseArray> for SparseEncoding {
    fn as_null_array<'a>(&self, array: &'a SparseArray) -> Option<&'a dyn NullArrayTrait> {
        Some(array)
    }

    fn as_bool_array<'a>(&self, array: &'a SparseArray) -> Option<&'a dyn BoolArrayTrait> {
        Some(array)
    }

    fn as_primitive_array<'a>(
        &self,
        array: &'a SparseArray,
    ) -> Option<&'a dyn PrimitiveArrayTrait> {
        Some(array)
    }

    fn as_utf8_array<'a>(&self, array: &'a SparseArray) -> Option<&'a dyn Utf8ArrayTrait> {
        Some(array)
    }

    fn as_binary_array<'a>(&self, array: &'a SparseArray) -> Option<&'a dyn BinaryArrayTrait> {
        Some(array)
    }

    fn as_struct_array<'a>(&self, array: &'a SparseArray) -> Option<&'a dyn StructArrayTrait> {
        Some(array)
    }

    fn as_list_array<'a>(&self, array: &'a SparseArray) -> Option<&'a dyn ListArrayTrait> {
        Some(array)
    }

    fn as_extension_array<'a>(
        &self,
        array: &'a SparseArray,
    ) -> Option<&'a dyn ExtensionArrayTrait> {
        Some(array)
    }
}

impl NullArrayTrait for SparseArray {}

impl BoolArrayTrait for SparseArray {}

impl PrimitiveArrayTrait for SparseArray {}

impl Utf8ArrayTrait for SparseArray {}

impl BinaryArrayTrait for SparseArray {}

impl StructArrayTrait for SparseArray {
    fn field(&self, idx: usize) -> Option<ArrayData> {
        let values = self.values().as_struct_array().and_then(|s| s.field(idx))?;
        let scalar = StructScalar::try_from(&self.fill_scalar())
            .ok()?
            .field_by_idx(idx)?;

        Some(
            SparseArray::try_new_with_offset(
                self.indices(),
                values,
                self.len(),
                self.indices_offset(),
                scalar,
            )
            .ok()?
            .into_array(),
        )
    }

    fn project(&self, projection: &[Field]) -> VortexResult<ArrayData> {
        let values = self
            .values()
            .as_struct_array()
            .ok_or_else(|| vortex_err!("Chunk was not a StructArray"))?
            .project(projection)?;
        let scalar = StructScalar::try_from(&self.fill_scalar())?.project(projection)?;

        SparseArray::try_new_with_offset(
            self.indices(),
            values,
            self.len(),
            self.indices_offset(),
            scalar,
        )
        .map(|a| a.into_array())
    }
}

impl ListArrayTrait for SparseArray {}

impl ExtensionArrayTrait for SparseArray {
    fn storage_data(&self) -> ArrayData {
        SparseArray::try_new_with_offset(
            self.indices(),
            self.values()
                .as_extension_array()
                .vortex_expect("Expected extension array")
                .storage_data(),
            self.len(),
            self.indices_offset(),
            self.fill_scalar(),
        )
        .vortex_expect("Failed to create new sparse array")
        .into_array()
    }
}

#[cfg(test)]
mod tests {
    use vortex_scalar::Scalar;

    use crate::array::{BoolArray, PrimitiveArray, SparseArray};
    use crate::compute::invert;
    use crate::{IntoArrayData, IntoArrayVariant};

    #[test]
    fn invert_bools_non_null_fill() {
        let sparse_bools = SparseArray::try_new(
            PrimitiveArray::from(vec![0u64]).into_array(),
            BoolArray::from_iter([false]).into_array(),
            2,
            Scalar::from(true),
        )
        .unwrap()
        .into_array();
        let inverted = invert(&sparse_bools).unwrap();
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
