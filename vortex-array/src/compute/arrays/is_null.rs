// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hasher;
use std::ops::Not;

use vortex_dtype::DType;
use vortex_dtype::Nullability::NonNullable;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_vector::{BoolVector, VectorOps};

use crate::execution::{BatchKernelRef, BindCtx, kernel};
use crate::stats::{ArrayStats, StatsSetRef};
use crate::vtable::{ArrayVTable, NotSupported, OperatorVTable, VTable, VisitorVTable};
use crate::{
    ArrayBufferVisitor, ArrayChildVisitor, ArrayEq, ArrayHash, ArrayRef, EncodingId, EncodingRef,
    Precision, vtable,
};

vtable!(IsNull);

#[derive(Debug, Clone)]
pub struct IsNullArray {
    child: ArrayRef,
    stats: ArrayStats,
}

impl IsNullArray {
    /// Create a new is_null array.
    pub fn new(child: ArrayRef) -> Self {
        Self {
            child,
            stats: ArrayStats::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct IsNullEncoding;

impl VTable for IsNullVTable {
    type Array = IsNullArray;
    type Encoding = IsNullEncoding;
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
        EncodingId::from("vortex.is_null")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::from(IsNullEncoding.as_ref())
    }
}

impl ArrayVTable<IsNullVTable> for IsNullVTable {
    fn len(array: &IsNullArray) -> usize {
        array.child.len()
    }

    fn dtype(_array: &IsNullArray) -> &DType {
        &DType::Bool(NonNullable)
    }

    fn stats(array: &IsNullArray) -> StatsSetRef<'_> {
        array.stats.to_ref(array.as_ref())
    }

    fn array_hash<H: Hasher>(array: &IsNullArray, state: &mut H, precision: Precision) {
        array.child.array_hash(state, precision);
    }

    fn array_eq(array: &IsNullArray, other: &IsNullArray, precision: Precision) -> bool {
        array.child.array_eq(&other.child, precision)
    }
}

impl VisitorVTable<IsNullVTable> for IsNullVTable {
    fn visit_buffers(_array: &IsNullArray, _visitor: &mut dyn ArrayBufferVisitor) {
        // No buffers
    }

    fn visit_children(array: &IsNullArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("child", array.child.as_ref());
    }
}

impl OperatorVTable<IsNullVTable> for IsNullVTable {
    fn bind(
        array: &IsNullArray,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        let child = ctx.bind(&array.child, selection)?;
        Ok(kernel(move || {
            let child = child.execute()?;
            let is_null = child.validity().not().to_bit_buffer();
            Ok(BoolVector::new(is_null, Mask::AllTrue(child.len())).into())
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Not;

    use vortex_buffer::{bitbuffer, buffer};
    use vortex_error::VortexResult;
    use vortex_vector::VectorOps;

    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::compute::arrays::is_null::IsNullArray;
    use crate::validity::Validity;

    #[test]
    fn test_is_null() -> VortexResult<()> {
        let validity = bitbuffer![1 0 1];
        let array = PrimitiveArray::new(
            buffer![0, 1, 2],
            Validity::Array(validity.clone().into_array()),
        )
        .into_array();

        let result = IsNullArray::new(array).execute()?.into_bool();
        assert!(result.validity().all_true());
        assert_eq!(result.bits(), &validity.not());

        Ok(())
    }
}
