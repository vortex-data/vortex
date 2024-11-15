use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::{ExtScalar, Scalar};

use crate::array::extension::ExtensionArray;
use crate::array::ConstantArray;
use crate::compute::unary::{scalar_at, scalar_at_unchecked, CastFn, ScalarAtFn};
use crate::compute::{
    compare, slice, take, ArrayCompute, MaybeCompareFn, Operator, SliceFn, TakeFn,
};
use crate::variants::ExtensionArrayTrait;
use crate::{ArrayDType, ArrayData, IntoArrayData};

impl ArrayCompute for ExtensionArray {
    fn cast(&self) -> Option<&dyn CastFn> {
        // It's not possible to cast an extension array to another type.
        // TODO(ngates): we should allow some extension arrays to implement a callback
        //  to support this
        None
    }

    fn compare(&self, other: &ArrayData, operator: Operator) -> Option<VortexResult<ArrayData>> {
        MaybeCompareFn::maybe_compare(self, other, operator)
    }

    fn scalar_at(&self) -> Option<&dyn ScalarAtFn> {
        Some(self)
    }

    fn slice(&self) -> Option<&dyn SliceFn> {
        Some(self)
    }

    fn take(&self) -> Option<&dyn TakeFn> {
        Some(self)
    }
}

impl MaybeCompareFn for ExtensionArray {
    fn maybe_compare(
        &self,
        other: &ArrayData,
        operator: Operator,
    ) -> Option<VortexResult<ArrayData>> {
        if let Ok(const_ext) = ConstantArray::try_from(other) {
            let scalar_ext = ExtScalar::try_new(const_ext.dtype(), const_ext.scalar_value())
                .vortex_expect("Expected ExtScalar");
            let const_storage = ConstantArray::new(
                Scalar::new(self.storage().dtype().clone(), scalar_ext.value().clone()),
                const_ext.len(),
            );

            return Some(compare(self.storage(), const_storage, operator));
        }

        if let Ok(rhs_ext) = ExtensionArray::try_from(other) {
            return Some(compare(self.storage(), rhs_ext.storage(), operator));
        }

        None
    }
}

impl ScalarAtFn for ExtensionArray {
    fn scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        Ok(Scalar::extension(
            self.ext_dtype().clone(),
            scalar_at(self.storage(), index)?.into_value(),
        ))
    }

    fn scalar_at_unchecked(&self, index: usize) -> Scalar {
        Scalar::extension(
            self.ext_dtype().clone(),
            scalar_at_unchecked(self.storage(), index).into_value(),
        )
    }
}

impl SliceFn for ExtensionArray {
    fn slice(&self, start: usize, stop: usize) -> VortexResult<ArrayData> {
        Ok(Self::new(
            self.ext_dtype().clone(),
            slice(self.storage(), start, stop)?,
        )
        .into_array())
    }
}

impl TakeFn for ExtensionArray {
    fn take(&self, indices: &ArrayData) -> VortexResult<ArrayData> {
        Ok(Self::new(self.ext_dtype().clone(), take(self.storage(), indices)?).into_array())
    }
}
