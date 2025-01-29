use vortex_array::variants::{
    BinaryArrayTrait, BoolArrayTrait, ExtensionArrayTrait, ListArrayTrait, NullArrayTrait,
    PrimitiveArrayTrait, StructArrayTrait, Utf8ArrayTrait,
};
use vortex_array::vtable::VariantsVTable;
use vortex_dtype::FieldName;
use vortex_error::{vortex_err, VortexExpect, VortexResult};
use vortex_scalar::StructScalar;

use crate::{ArrayData, ArrayLen, IntoArrayData, SparseArray, SparseEncoding};

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
    fn maybe_null_field_by_idx(&self, idx: usize) -> Option<ArrayData> {
        let new_patches = self
            .patches()
            .map_values_opt(|values| {
                values
                    .as_struct_array()
                    .and_then(|s| s.maybe_null_field_by_idx(idx))
            })
            .vortex_expect("field array length should equal struct array length")?;
        let scalar = StructScalar::try_from(&self.fill_scalar())
            .ok()?
            .field_by_idx(idx)?;

        Some(
            SparseArray::try_new_from_patches(
                new_patches,
                self.len(),
                self.indices_offset(),
                scalar,
            )
            .ok()?
            .into_array(),
        )
    }

    fn project(&self, projection: &[FieldName]) -> VortexResult<ArrayData> {
        let new_patches = self.patches().map_values(|values| {
            values
                .as_struct_array()
                .ok_or_else(|| vortex_err!("Chunk was not a StructArray"))?
                .project(projection)
        })?;
        let scalar = StructScalar::try_from(&self.fill_scalar())?.project(projection)?;

        SparseArray::try_new_from_patches(new_patches, self.len(), self.indices_offset(), scalar)
            .map(IntoArrayData::into_array)
    }
}

impl ListArrayTrait for SparseArray {}

impl ExtensionArrayTrait for SparseArray {
    fn storage_data(&self) -> ArrayData {
        SparseArray::try_new_from_patches(
            self.patches()
                .map_values(|values| {
                    Ok(values
                        .as_extension_array()
                        .vortex_expect("Expected extension array")
                        .storage_data())
                })
                .vortex_expect("as_extension_array preserves the length"),
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
    use vortex_array::array::BoolArray;
    use vortex_array::compute::invert;
    use vortex_array::{IntoArrayData, IntoArrayVariant};
    use vortex_buffer::buffer;
    use vortex_scalar::Scalar;

    use crate::SparseArray;

    #[test]
    fn invert_bools_non_null_fill() {
        let sparse_bools = SparseArray::try_new(
            buffer![0u64].into_array(),
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
