// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hasher;

use vortex_dtype::DType;
use vortex_dtype::Nullability::NonNullable;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_vector::VectorOps;
use vortex_vector::bool::BoolVector;

use crate::execution::{BatchKernelRef, BindCtx, kernel};
use crate::stats::{ArrayStats, StatsSetRef};
use crate::vtable::{ArrayVTable, NotSupported, OperatorVTable, VTable, VisitorVTable};
use crate::{
    ArrayBufferVisitor, ArrayChildVisitor, ArrayEq, ArrayHash, ArrayRef, EncodingId, EncodingRef,
    Precision, vtable,
};

vtable!(IsNotNull);

#[derive(Debug, Clone)]
pub struct IsNotNullArray {
    child: ArrayRef,
    stats: ArrayStats,
}

impl IsNotNullArray {
    /// Create a new is_not_null array.
    pub fn new(child: ArrayRef) -> Self {
        Self {
            child,
            stats: ArrayStats::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct IsNotNullEncoding;

impl VTable for IsNotNullVTable {
    type Array = IsNotNullArray;
    type Encoding = IsNotNullEncoding;
    type ArrayVTable = Self;
    type CanonicalVTable = NotSupported;
    type OperationsVTable = NotSupported;
    type ValidityVTable = NotSupported;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type SerdeVTable = NotSupported;
    type OperatorVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::from("vortex.is_not_null")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::from(IsNotNullEncoding.as_ref())
    }
}

impl ArrayVTable<IsNotNullVTable> for IsNotNullVTable {
    fn len(array: &IsNotNullArray) -> usize {
        array.child.len()
    }

    fn dtype(_array: &IsNotNullArray) -> &DType {
        &DType::Bool(NonNullable)
    }

    fn stats(array: &IsNotNullArray) -> StatsSetRef<'_> {
        array.stats.to_ref(array.as_ref())
    }

    fn array_hash<H: Hasher>(array: &IsNotNullArray, state: &mut H, precision: Precision) {
        array.child.array_hash(state, precision);
    }

    fn array_eq(array: &IsNotNullArray, other: &IsNotNullArray, precision: Precision) -> bool {
        array.child.array_eq(&other.child, precision)
    }
}

impl VisitorVTable<IsNotNullVTable> for IsNotNullVTable {
    fn visit_buffers(_array: &IsNotNullArray, _visitor: &mut dyn ArrayBufferVisitor) {
        // No buffers
    }

    fn visit_children(array: &IsNotNullArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("child", array.child.as_ref());
    }
}

impl OperatorVTable<IsNotNullVTable> for IsNotNullVTable {
    fn bind(
        array: &IsNotNullArray,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        let child = ctx.bind(&array.child, selection)?;
        Ok(kernel(move || {
            let child = child.execute()?;
            let is_null = child.validity().to_bit_buffer();
            Ok(BoolVector::new(is_null, Mask::AllTrue(child.len())).into())
        }))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::{bitbuffer, buffer};
    use vortex_error::VortexResult;
    use vortex_vector::VectorOps;

    use super::IsNotNullArray;
    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::validity::Validity;

    #[test]
    fn test_is_null() -> VortexResult<()> {
        let validity = bitbuffer![1 0 1];
        let array = PrimitiveArray::new(
            buffer![0, 1, 2],
            Validity::Array(validity.clone().into_array()),
        )
        .into_array();

        let result = IsNotNullArray::new(array).execute()?.into_bool();
        assert!(result.validity().all_true());
        assert_eq!(result.bits(), &validity);

        Ok(())
    }
}
