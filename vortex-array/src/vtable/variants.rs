use vortex_error::{VortexError, VortexExpect};

use crate::encoding::Encoding;
use crate::variants::{
    BinaryArrayTrait, BoolArrayTrait, ExtensionArrayTrait, ListArrayTrait, NullArrayTrait,
    PrimitiveArrayTrait, StructArrayTrait, Utf8ArrayTrait,
};
use crate::ArrayData;

/// An Array encoding must declare which DTypes it can be downcast into.
pub trait VariantsVTable<Array> {
    fn as_null_array<'a>(&self, _array: &'a Array) -> Option<&'a dyn NullArrayTrait> {
        None
    }

    fn as_bool_array<'a>(&self, _array: &'a Array) -> Option<&'a dyn BoolArrayTrait> {
        None
    }

    fn as_primitive_array<'a>(&self, _array: &'a Array) -> Option<&'a dyn PrimitiveArrayTrait> {
        None
    }

    fn as_utf8_array<'a>(&self, _array: &'a Array) -> Option<&'a dyn Utf8ArrayTrait> {
        None
    }

    fn as_binary_array<'a>(&self, _array: &'a Array) -> Option<&'a dyn BinaryArrayTrait> {
        None
    }

    fn as_struct_array<'a>(&self, _array: &'a Array) -> Option<&'a dyn StructArrayTrait> {
        None
    }

    fn as_list_array<'a>(&self, _array: &'a Array) -> Option<&'a dyn ListArrayTrait> {
        None
    }

    fn as_extension_array<'a>(&self, _array: &'a Array) -> Option<&'a dyn ExtensionArrayTrait> {
        None
    }
}

impl<E: Encoding> VariantsVTable<ArrayData> for E
where
    E: VariantsVTable<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn as_null_array<'a>(&self, array: &'a ArrayData) -> Option<&'a dyn NullArrayTrait> {
        let (array_ref, encoding) = array
            .try_downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        VariantsVTable::as_null_array(encoding, array_ref)
    }

    fn as_bool_array<'a>(&self, array: &'a ArrayData) -> Option<&'a dyn BoolArrayTrait> {
        let (array_ref, encoding) = array
            .try_downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        VariantsVTable::as_bool_array(encoding, array_ref)
    }

    fn as_primitive_array<'a>(&self, array: &'a ArrayData) -> Option<&'a dyn PrimitiveArrayTrait> {
        let (array_ref, encoding) = array
            .try_downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        VariantsVTable::as_primitive_array(encoding, array_ref)
    }

    fn as_utf8_array<'a>(&self, array: &'a ArrayData) -> Option<&'a dyn Utf8ArrayTrait> {
        let (array_ref, encoding) = array
            .try_downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        VariantsVTable::as_utf8_array(encoding, array_ref)
    }

    fn as_binary_array<'a>(&self, array: &'a ArrayData) -> Option<&'a dyn BinaryArrayTrait> {
        let (array_ref, encoding) = array
            .try_downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        VariantsVTable::as_binary_array(encoding, array_ref)
    }

    fn as_struct_array<'a>(&self, array: &'a ArrayData) -> Option<&'a dyn StructArrayTrait> {
        let (array_ref, encoding) = array
            .try_downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        VariantsVTable::as_struct_array(encoding, array_ref)
    }

    fn as_list_array<'a>(&self, array: &'a ArrayData) -> Option<&'a dyn ListArrayTrait> {
        let (array_ref, encoding) = array
            .try_downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        VariantsVTable::as_list_array(encoding, array_ref)
    }

    fn as_extension_array<'a>(&self, array: &'a ArrayData) -> Option<&'a dyn ExtensionArrayTrait> {
        let (array_ref, encoding) = array
            .try_downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        VariantsVTable::as_extension_array(encoding, array_ref)
    }
}
