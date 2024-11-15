use std::iter;
use std::sync::Arc;

use vortex_dtype::field::Field;
use vortex_dtype::{DType, PType};
use vortex_error::{VortexError, VortexExpect as _, VortexResult};
use vortex_scalar::{Scalar, ScalarValue, StructScalar};

use crate::array::constant::ConstantArray;
use crate::iter::{Accessor, AccessorRef};
use crate::validity::{ArrayValidity, Validity};
use crate::variants::{
    ArrayVariants, BinaryArrayTrait, BoolArrayTrait, ExtensionArrayTrait, ListArrayTrait,
    NullArrayTrait, PrimitiveArrayTrait, StructArrayTrait, Utf8ArrayTrait,
};
use crate::{ArrayDType, ArrayData, IntoArrayData};

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
        let value = self.scalar_value().as_bool()?;
        match value {
            None => Ok(self.clone().into_array()),
            Some(b) => Ok(ConstantArray::new(!b, self.len()).into_array()),
        }
    }

    fn maybe_null_indices_iter(&self) -> Box<dyn Iterator<Item = usize>> {
        let value = self
            .scalar_value()
            .as_bool()
            .vortex_expect("Failed to get bool value from constant array");
        if value.unwrap_or(false) {
            Box::new(0..self.len())
        } else {
            Box::new(iter::empty())
        }
    }

    fn maybe_null_slices_iter(&self) -> Box<dyn Iterator<Item = (usize, usize)>> {
        // Must be a boolean scalar
        let value = self
            .scalar_value()
            .as_bool()
            .vortex_expect("Failed to get bool value from constant array");

        if value.unwrap_or(false) {
            Box::new(iter::once((0, self.len())))
        } else {
            Box::new(iter::empty())
        }
    }
}

impl<T> Accessor<T> for ConstantArray
where
    T: Clone,
    T: TryFrom<ScalarValue, Error = VortexError>,
{
    fn array_len(&self) -> usize {
        self.len()
    }

    fn is_valid(&self, index: usize) -> bool {
        ArrayValidity::is_valid(self, index)
    }

    fn value_unchecked(&self, _index: usize) -> T {
        T::try_from(self.scalar_value().clone()).vortex_expect("Failed to convert scalar to value")
    }

    fn array_validity(&self) -> Validity {
        if self.scalar_value().is_null() {
            Validity::AllInvalid
        } else {
            Validity::AllValid
        }
    }
}

impl PrimitiveArrayTrait for ConstantArray {
    fn f32_accessor(&self) -> Option<AccessorRef<f32>> {
        match self.dtype() {
            DType::Primitive(PType::F32, _) => Some(Arc::new(self.clone())),
            _ => None,
        }
    }

    fn f64_accessor(&self) -> Option<AccessorRef<f64>> {
        match self.dtype() {
            DType::Primitive(PType::F64, _) => Some(Arc::new(self.clone())),
            _ => None,
        }
    }

    fn u8_accessor(&self) -> Option<AccessorRef<u8>> {
        match self.dtype() {
            DType::Primitive(PType::U8, _) => Some(Arc::new(self.clone())),
            _ => None,
        }
    }

    fn u16_accessor(&self) -> Option<AccessorRef<u16>> {
        match self.dtype() {
            DType::Primitive(PType::U16, _) => Some(Arc::new(self.clone())),
            _ => None,
        }
    }

    fn u32_accessor(&self) -> Option<AccessorRef<u32>> {
        match self.dtype() {
            DType::Primitive(PType::U32, _) => Some(Arc::new(self.clone())),
            _ => None,
        }
    }

    fn u64_accessor(&self) -> Option<AccessorRef<u64>> {
        match self.dtype() {
            DType::Primitive(PType::U64, _) => Some(Arc::new(self.clone())),
            _ => None,
        }
    }

    fn i8_accessor(&self) -> Option<AccessorRef<i8>> {
        match self.dtype() {
            DType::Primitive(PType::I8, _) => Some(Arc::new(self.clone())),
            _ => None,
        }
    }

    fn i16_accessor(&self) -> Option<AccessorRef<i16>> {
        match self.dtype() {
            DType::Primitive(PType::I16, _) => Some(Arc::new(self.clone())),
            _ => None,
        }
    }

    fn i32_accessor(&self) -> Option<AccessorRef<i32>> {
        match self.dtype() {
            DType::Primitive(PType::I32, _) => Some(Arc::new(self.clone())),
            _ => None,
        }
    }

    fn i64_accessor(&self) -> Option<AccessorRef<i64>> {
        match self.dtype() {
            DType::Primitive(PType::I64, _) => Some(Arc::new(self.clone())),
            _ => None,
        }
    }
}

impl Utf8ArrayTrait for ConstantArray {}

impl BinaryArrayTrait for ConstantArray {}

impl StructArrayTrait for ConstantArray {
    fn field(&self, idx: usize) -> Option<ArrayData> {
        StructScalar::try_new(self.dtype(), self.scalar_value())
            .ok()?
            .field_by_idx(idx)
            .map(|scalar| ConstantArray::new(scalar, self.len()).into_array())
    }

    fn project(&self, projection: &[Field]) -> VortexResult<ArrayData> {
        Ok(ConstantArray::new(
            StructScalar::try_new(self.dtype(), self.scalar_value())?.project(projection)?,
            self.len(),
        )
        .into_array())
    }
}

impl ListArrayTrait for ConstantArray {}

impl ExtensionArrayTrait for ConstantArray {
    fn storage_data(&self) -> ArrayData {
        let storage_dtype = self.ext_dtype().storage_dtype().clone();
        ConstantArray::new(
            Scalar::new(storage_dtype, self.scalar_value().clone()),
            self.len(),
        )
        .into_array()
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use vortex_dtype::Nullability;
    use vortex_scalar::Scalar;

    use crate::array::constant::ConstantArray;
    use crate::variants::BoolArrayTrait;

    #[test]
    fn constant_iter_true_test() {
        let arr = ConstantArray::new(Scalar::bool(true, Nullability::NonNullable), 3);
        assert_eq!(vec![0, 1, 2], arr.maybe_null_indices_iter().collect_vec());
        assert_eq!(vec![(0, 3)], arr.maybe_null_slices_iter().collect_vec());
    }

    #[test]
    fn constant_iter_false_test() {
        let arr = ConstantArray::new(Scalar::bool(false, Nullability::NonNullable), 3);
        assert_eq!(0, arr.maybe_null_indices_iter().collect_vec().len());
        assert_eq!(0, arr.maybe_null_slices_iter().collect_vec().len());
    }
}
