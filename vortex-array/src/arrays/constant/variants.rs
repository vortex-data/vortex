use vortex_dtype::FieldName;
use vortex_error::{VortexError, VortexExpect as _, VortexResult};
use vortex_scalar::Scalar;

use crate::arrays::constant::ConstantArray;
use crate::arrays::ConstantEncoding;
use crate::iter::Accessor;
use crate::variants::{
    BinaryArrayTrait, BoolArrayTrait, ExtensionArrayTrait, ListArrayTrait, NullArrayTrait,
    PrimitiveArrayTrait, StructArrayTrait, Utf8ArrayTrait,
};
use crate::vtable::VariantsVTable;
use crate::{Array, IntoArray};

/// Constant arrays support all DTypes
impl VariantsVTable<ConstantArray> for ConstantEncoding {
    fn as_null_array<'a>(&self, array: &'a ConstantArray) -> Option<&'a dyn NullArrayTrait> {
        Some(array)
    }

    fn as_bool_array<'a>(&self, array: &'a ConstantArray) -> Option<&'a dyn BoolArrayTrait> {
        Some(array)
    }

    fn as_primitive_array<'a>(
        &self,
        array: &'a ConstantArray,
    ) -> Option<&'a dyn PrimitiveArrayTrait> {
        Some(array)
    }

    fn as_utf8_array<'a>(&self, array: &'a ConstantArray) -> Option<&'a dyn Utf8ArrayTrait> {
        Some(array)
    }

    fn as_binary_array<'a>(&self, array: &'a ConstantArray) -> Option<&'a dyn BinaryArrayTrait> {
        Some(array)
    }

    fn as_struct_array<'a>(&self, array: &'a ConstantArray) -> Option<&'a dyn StructArrayTrait> {
        Some(array)
    }

    fn as_list_array<'a>(&self, array: &'a ConstantArray) -> Option<&'a dyn ListArrayTrait> {
        Some(array)
    }

    fn as_extension_array<'a>(
        &self,
        array: &'a ConstantArray,
    ) -> Option<&'a dyn ExtensionArrayTrait> {
        Some(array)
    }
}

impl NullArrayTrait for ConstantArray {}

impl BoolArrayTrait for ConstantArray {}

impl<T> Accessor<T> for ConstantArray
where
    T: Clone,
    T: TryFrom<Scalar, Error = VortexError>,
{
    fn value_unchecked(&self, _index: usize) -> T {
        T::try_from(self.scalar()).vortex_expect("Failed to convert scalar to value")
    }
}

impl PrimitiveArrayTrait for ConstantArray {}

impl Utf8ArrayTrait for ConstantArray {}

impl BinaryArrayTrait for ConstantArray {}

impl StructArrayTrait for ConstantArray {
    fn maybe_null_field_by_idx(&self, idx: usize) -> VortexResult<Array> {
        self.scalar()
            .as_struct()
            .field_by_idx(idx)
            .map(|scalar| ConstantArray::new(scalar, self.len()).into_array())
    }

    fn project(&self, projection: &[FieldName]) -> VortexResult<Array> {
        Ok(
            ConstantArray::new(self.scalar().as_struct().project(projection)?, self.len())
                .into_array(),
        )
    }
}

impl ListArrayTrait for ConstantArray {}

impl ExtensionArrayTrait for ConstantArray {
    fn storage_data(&self) -> Array {
        ConstantArray::new(self.scalar().as_extension().storage(), self.len()).into_array()
    }
}
