use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::{ExtScalar, Scalar};

use crate::array::extension::ExtensionArray;
use crate::array::{ConstantArray, ExtensionEncoding};
use crate::compute::unary::{scalar_at, CastFn, ScalarAtFn};
use crate::compute::{
    compare, slice, take, ArrayCompute, ComputeVTable, MaybeCompareFn, Operator, SliceFn, TakeFn,
    TakeOptions,
};
use crate::variants::ExtensionArrayTrait;
use crate::{ArrayDType, ArrayData, ArrayLen, IntoArrayData};

impl ArrayCompute for ExtensionArray {
    fn compare(&self, other: &ArrayData, operator: Operator) -> Option<VortexResult<ArrayData>> {
        MaybeCompareFn::maybe_compare(self, other, operator)
    }
}

impl ComputeVTable for ExtensionEncoding {
    fn cast_fn(&self) -> Option<&dyn CastFn<ArrayData>> {
        // It's not possible to cast an extension array to another type.
        // TODO(ngates): we should allow some extension arrays to implement a callback
        //  to support this
        None
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<ArrayData>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<ArrayData>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<ArrayData>> {
        Some(self)
    }
}

impl MaybeCompareFn for ExtensionArray {
    fn maybe_compare(
        &self,
        other: &ArrayData,
        operator: Operator,
    ) -> Option<VortexResult<ArrayData>> {
        if let Some(const_ext) = other.as_constant() {
            let scalar_ext = ExtScalar::try_new(const_ext.dtype(), const_ext.value())
                .vortex_expect("Expected ExtScalar");
            let const_storage = ConstantArray::new(
                Scalar::new(self.storage().dtype().clone(), scalar_ext.value().clone()),
                self.len(),
            );

            return Some(compare(self.storage(), const_storage, operator));
        }

        // TODO(ngates): do not use try_from to test for encoding.
        if let Ok(rhs_ext) = ExtensionArray::try_from(other.clone()) {
            return Some(compare(self.storage(), rhs_ext.storage(), operator));
        }

        None
    }
}

impl ScalarAtFn<ExtensionArray> for ExtensionEncoding {
    fn scalar_at(&self, array: &ExtensionArray, index: usize) -> VortexResult<Scalar> {
        Ok(Scalar::extension(
            array.ext_dtype().clone(),
            scalar_at(array.storage(), index)?.into_value(),
        ))
    }
}

impl SliceFn<ExtensionArray> for ExtensionEncoding {
    fn slice(&self, array: &ExtensionArray, start: usize, stop: usize) -> VortexResult<ArrayData> {
        Ok(ExtensionArray::new(
            array.ext_dtype().clone(),
            slice(array.storage(), start, stop)?,
        )
        .into_array())
    }
}

impl TakeFn<ExtensionArray> for ExtensionEncoding {
    fn take(
        &self,
        array: &ExtensionArray,
        indices: &ArrayData,
        options: TakeOptions,
    ) -> VortexResult<ArrayData> {
        Ok(ExtensionArray::new(
            array.ext_dtype().clone(),
            take(array.storage(), indices, options)?,
        )
        .into_array())
    }
}
