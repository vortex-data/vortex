use vortex_dtype::field::Field;
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexExpect as _, VortexResult};
use vortex_scalar::Scalar;

use crate::array::constant::ConstantArray;
use crate::iter::Accessor;
use crate::validity::{ArrayValidity, Validity};
use crate::variants::{
    ArrayVariants, BinaryArrayTrait, BoolArrayTrait, ExtensionArrayTrait, ListArrayTrait,
    NullArrayTrait, PrimitiveArrayTrait, StructArrayTrait, Utf8ArrayTrait,
};
use crate::{ArrayDType, ArrayData, ArrayLen, IntoArrayData, ToArrayData};

/// Constant arrays support all DTypes
impl ArrayVariants for ConstantArray {
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

impl NullArrayTrait for ConstantArray {}

impl BoolArrayTrait for ConstantArray {
    fn invert(&self) -> VortexResult<ArrayData> {
        match self.scalar().as_bool().value() {
            None => Ok(self.to_array()),
            Some(b) => Ok(ConstantArray::new(!b, self.len()).into_array()),
        }
    }
}

impl<T> Accessor<T> for ConstantArray
where
    T: Clone,
    T: TryFrom<Scalar, Error = VortexError>,
{
    fn array_len(&self) -> usize {
        self.len()
    }

    fn is_valid(&self, index: usize) -> bool {
        ArrayValidity::is_valid(self, index)
    }

    fn value_unchecked(&self, _index: usize) -> T {
        T::try_from(self.scalar()).vortex_expect("Failed to convert scalar to value")
    }

    fn array_validity(&self) -> Validity {
        if self.scalar().is_null() {
            Validity::AllInvalid
        } else {
            Validity::AllValid
        }
    }
}

impl PrimitiveArrayTrait for ConstantArray {}

impl Utf8ArrayTrait for ConstantArray {}

impl BinaryArrayTrait for ConstantArray {}

impl StructArrayTrait for ConstantArray {
    fn field(&self, idx: usize) -> Option<ArrayData> {
        self.scalar()
            .as_struct()
            .field_by_idx(idx)
            .map(|scalar| ConstantArray::new(scalar, self.len()).into_array())
    }

    fn project(&self, projection: &[Field]) -> VortexResult<ArrayData> {
        Ok(
            ConstantArray::new(self.scalar().as_struct().project(projection)?, self.len())
                .into_array(),
        )
    }
}

impl ListArrayTrait for ConstantArray {}

impl ExtensionArrayTrait for ConstantArray {
    fn storage_data(&self) -> ArrayData {
        ConstantArray::new(self.scalar().as_extension().storage(), self.len()).into_array()
    }
}
