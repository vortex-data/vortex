use vortex_dtype::FieldName;
use vortex_error::VortexResult;
use vortex_scalar::PValue;

use crate::arrays::constant::ConstantArray;
use crate::variants::{
    BinaryArrayTrait, BoolArrayTrait, DecimalArrayTrait, ExtensionArrayTrait, ListArrayTrait,
    NullArrayTrait, PrimitiveArrayTrait, StructArrayTrait, Utf8ArrayTrait,
};
use crate::{Array, ArrayRef, ArrayVariantsImpl};

/// Constant arrays support all DTypes
impl ArrayVariantsImpl for ConstantArray {
    fn _as_null_typed(&self) -> Option<&dyn NullArrayTrait> {
        Some(self)
    }

    fn _as_bool_typed(&self) -> Option<&dyn BoolArrayTrait> {
        Some(self)
    }

    fn _as_primitive_typed(&self) -> Option<&dyn PrimitiveArrayTrait> {
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

    fn _as_decimal_typed(&self) -> Option<&dyn DecimalArrayTrait> {
        Some(self)
    }
}

impl NullArrayTrait for ConstantArray {}

impl BoolArrayTrait for ConstantArray {}

impl PrimitiveArrayTrait for ConstantArray {
    fn value(&self, _idx: usize) -> Option<PValue> {
        self.scalar().as_primitive().pvalue()
    }
}

impl Utf8ArrayTrait for ConstantArray {}

impl BinaryArrayTrait for ConstantArray {}

impl DecimalArrayTrait for ConstantArray {}

impl StructArrayTrait for ConstantArray {
    fn maybe_null_field_by_idx(&self, idx: usize) -> VortexResult<ArrayRef> {
        self.scalar()
            .as_struct()
            .field_by_idx(idx)
            .map(|scalar| ConstantArray::new(scalar, self.len()).into_array())
    }

    fn project(&self, projection: &[FieldName]) -> VortexResult<ArrayRef> {
        Ok(
            ConstantArray::new(self.scalar().as_struct().project(projection)?, self.len())
                .into_array(),
        )
    }
}

impl ListArrayTrait for ConstantArray {}

impl ExtensionArrayTrait for ConstantArray {
    fn storage_data(&self) -> ArrayRef {
        ConstantArray::new(self.scalar().as_extension().storage(), self.len()).into_array()
    }
}
