use vortex_array::variants::{
    BinaryArrayTrait, BoolArrayTrait, DecimalArrayTrait, ExtensionArrayTrait, ListArrayTrait,
    NullArrayTrait, PrimitiveArrayTrait, StructArrayTrait, Utf8ArrayTrait,
};
use vortex_array::{Array, ArrayVariants, ArrayVariantsImpl};
use vortex_dtype::FieldName;
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_scalar::StructScalar;

use crate::{ArrayRef, SparseArray};

/// Sparse arrays support all DTypes
impl ArrayVariantsImpl for SparseArray {
    fn _as_null_typed(&self) -> Option<&dyn NullArrayTrait> {
        Some(self)
    }

    fn _as_bool_typed(&self) -> Option<&dyn BoolArrayTrait> {
        Some(self)
    }

    fn _as_primitive_typed(&self) -> Option<&dyn PrimitiveArrayTrait> {
        Some(self)
    }

    fn _as_decimal_typed(&self) -> Option<&dyn DecimalArrayTrait> {
        Some(self)
    }

    fn _as_utf8_typed(&self) -> Option<&dyn Utf8ArrayTrait> {
        Some(self)
    }

    fn _as_binary_typed(&self) -> Option<&dyn BinaryArrayTrait> {
        Some(self)
    }

    fn _as_struct_typed(&self) -> Option<&dyn StructArrayTrait> {
        Some(self)
    }

    fn _as_list_typed(&self) -> Option<&dyn ListArrayTrait> {
        Some(self)
    }

    fn _as_extension_typed(&self) -> Option<&dyn ExtensionArrayTrait> {
        Some(self)
    }
}

impl NullArrayTrait for SparseArray {}

impl BoolArrayTrait for SparseArray {}

impl PrimitiveArrayTrait for SparseArray {}

impl Utf8ArrayTrait for SparseArray {}

impl BinaryArrayTrait for SparseArray {}

impl DecimalArrayTrait for SparseArray {}

impl StructArrayTrait for SparseArray {
    fn maybe_null_field_by_idx(&self, idx: usize) -> VortexResult<ArrayRef> {
        let new_patches = self.patches().clone().map_values(|values| {
            values
                .as_struct_typed()
                .vortex_expect("Expected struct array")
                .maybe_null_field_by_idx(idx)
        })?;
        let scalar = StructScalar::try_from(self.fill_scalar())?.field_by_idx(idx)?;

        Ok(SparseArray::try_new_from_patches(new_patches, scalar)?.into_array())
    }

    fn project(&self, projection: &[FieldName]) -> VortexResult<ArrayRef> {
        let new_patches = self.patches().clone().map_values(|values| {
            values
                .as_struct_typed()
                .ok_or_else(|| vortex_err!("Chunk was not a StructArray"))?
                .project(projection)
        })?;
        let scalar = StructScalar::try_from(self.fill_scalar())?.project(projection)?;

        Ok(SparseArray::try_new_from_patches(new_patches, scalar)?.into_array())
    }
}

impl ListArrayTrait for SparseArray {}

impl ExtensionArrayTrait for SparseArray {
    fn storage_data(&self) -> ArrayRef {
        SparseArray::try_new_from_patches(
            self.patches()
                .clone()
                .map_values(|values| {
                    Ok(values
                        .as_extension_typed()
                        .vortex_expect("Expected extension array")
                        .storage_data())
                })
                .vortex_expect("as_extension_array preserves the length"),
            self.fill_scalar().clone(),
        )
        .vortex_expect("Failed to create new sparse array")
        .into_array()
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::BoolArray;
    use vortex_array::compute::invert;
    use vortex_array::{Array, IntoArray, ToCanonical};
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
        .unwrap();
        let inverted = invert(&sparse_bools).unwrap();
        assert_eq!(
            inverted
                .to_bool()
                .unwrap()
                .boolean_buffer()
                .iter()
                .collect::<Vec<_>>(),
            vec![true, false]
        );
    }
}
