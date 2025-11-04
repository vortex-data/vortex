// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::{Hash, Hasher};
use std::sync::LazyLock;

use enum_map::{Enum, EnumMap, enum_map};
use vortex_buffer::ByteBuffer;
use vortex_compute::arithmetic::{
    Add, Arithmetic, CheckedArithmetic, CheckedOperator, Div, Mul, Operator, Sub,
};
use vortex_dtype::{DType, NativePType, PTypeDowncastExt, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_scalar::{PValue, Scalar};
use vortex_vector::primitive::PVector;

use crate::arrays::ConstantArray;
use crate::execution::{BatchKernelRef, BindCtx, kernel};
use crate::serde::ArrayChildren;
use crate::stats::{ArrayStats, StatsSetRef};
use crate::vtable::{
    ArrayVTable, NotSupported, OperatorVTable, SerdeVTable, VTable, VisitorVTable,
};
use crate::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayEq, ArrayHash, ArrayRef,
    DeserializeMetadata, EmptyMetadata, EncodingId, EncodingRef, IntoArray, Precision, vtable,
};

/// The set of operators supported by an arithmetic array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Enum)]
pub enum ArithmeticOperator {
    /// Addition - errors on overflow for integers.
    Add,
    /// Subtraction - errors on overflow for integers.
    Sub,
    /// Multiplication - errors on overflow for integers.
    Mul,
    /// Division - errors on division by zero for integers.
    Div,
}

vtable!(Arithmetic);

#[derive(Debug, Clone)]
pub struct ArithmeticArray {
    encoding: EncodingRef,
    lhs: ArrayRef,
    rhs: ArrayRef,
    stats: ArrayStats,
}

impl ArithmeticArray {
    /// Create a new arithmetic array.
    pub fn new(lhs: ArrayRef, rhs: ArrayRef, operator: ArithmeticOperator) -> Self {
        assert_eq!(
            lhs.len(),
            rhs.len(),
            "Arithmetic arrays require lhs and rhs to have the same length"
        );

        // TODO(ngates): should we automatically cast non-null to nullable if required?
        assert!(matches!(lhs.dtype(), DType::Primitive(..)));
        assert_eq!(lhs.dtype(), rhs.dtype());

        Self {
            encoding: ENCODINGS[operator].clone(),
            lhs,
            rhs,
            stats: ArrayStats::default(),
        }
    }

    /// Returns the operator of this logical array.
    pub fn operator(&self) -> ArithmeticOperator {
        self.encoding.as_::<ArithmeticVTable>().operator
    }
}

#[derive(Debug, Clone)]
pub struct ArithmeticEncoding {
    // We include the operator in the encoding so each operator is a different encoding ID.
    // This makes it easier for plugins to construct expressions and perform pushdown
    // optimizations.
    operator: ArithmeticOperator,
}

#[allow(clippy::mem_forget)]
static ENCODINGS: LazyLock<EnumMap<ArithmeticOperator, EncodingRef>> = LazyLock::new(|| {
    enum_map! {
        operator => ArithmeticEncoding { operator }.to_encoding(),
    }
});

impl VTable for ArithmeticVTable {
    type Array = ArithmeticArray;
    type Encoding = ArithmeticEncoding;
    type ArrayVTable = Self;
    type CanonicalVTable = NotSupported;
    type OperationsVTable = NotSupported;
    type ValidityVTable = NotSupported;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type SerdeVTable = Self;
    type OperatorVTable = Self;

    fn id(encoding: &Self::Encoding) -> EncodingId {
        match encoding.operator {
            ArithmeticOperator::Add => EncodingId::from("vortex.add"),
            ArithmeticOperator::Sub => EncodingId::from("vortex.sub"),
            ArithmeticOperator::Mul => EncodingId::from("vortex.mul"),
            ArithmeticOperator::Div => EncodingId::from("vortex.div"),
        }
    }

    fn encoding(array: &Self::Array) -> EncodingRef {
        array.encoding.clone()
    }
}

impl ArrayVTable<ArithmeticVTable> for ArithmeticVTable {
    fn len(array: &ArithmeticArray) -> usize {
        array.lhs.len()
    }

    fn dtype(array: &ArithmeticArray) -> &DType {
        array.lhs.dtype()
    }

    fn stats(array: &ArithmeticArray) -> StatsSetRef<'_> {
        array.stats.to_ref(array.as_ref())
    }

    fn array_hash<H: Hasher>(array: &ArithmeticArray, state: &mut H, precision: Precision) {
        array.lhs.array_hash(state, precision);
        array.rhs.array_hash(state, precision);
    }

    fn array_eq(array: &ArithmeticArray, other: &ArithmeticArray, precision: Precision) -> bool {
        array.lhs.array_eq(&other.lhs, precision) && array.rhs.array_eq(&other.rhs, precision)
    }
}

impl VisitorVTable<ArithmeticVTable> for ArithmeticVTable {
    fn visit_buffers(_array: &ArithmeticArray, _visitor: &mut dyn ArrayBufferVisitor) {
        // No buffers
    }

    fn visit_children(array: &ArithmeticArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("lhs", array.lhs.as_ref());
        visitor.visit_child("rhs", array.rhs.as_ref());
    }
}

impl SerdeVTable<ArithmeticVTable> for ArithmeticVTable {
    type Metadata = EmptyMetadata;

    fn metadata(_array: &ArithmeticArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(EmptyMetadata))
    }

    fn build(
        encoding: &ArithmeticEncoding,
        dtype: &DType,
        len: usize,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ArithmeticArray> {
        assert!(buffers.is_empty());

        Ok(ArithmeticArray::new(
            children.get(0, dtype, len)?,
            children.get(1, dtype, len)?,
            encoding.operator,
        ))
    }
}

impl OperatorVTable<ArithmeticVTable> for ArithmeticVTable {
    fn reduce_children(array: &ArithmeticArray) -> VortexResult<Option<ArrayRef>> {
        match (array.lhs.as_constant(), array.rhs.as_constant()) {
            // If both sides are constant, we compute the value now.
            (Some(lhs), Some(rhs)) => {
                let op: vortex_scalar::NumericOperator = match array.operator() {
                    ArithmeticOperator::Add => vortex_scalar::NumericOperator::Add,
                    ArithmeticOperator::Sub => vortex_scalar::NumericOperator::Sub,
                    ArithmeticOperator::Mul => vortex_scalar::NumericOperator::Mul,
                    ArithmeticOperator::Div => vortex_scalar::NumericOperator::Div,
                };
                let result = lhs
                    .as_primitive()
                    .checked_binary_numeric(&rhs.as_primitive(), op)
                    .ok_or_else(|| {
                        vortex_err!("Constant arithmetic operation resulted in overflow")
                    })?;
                return Ok(Some(
                    ConstantArray::new(Scalar::from(result), array.len()).into_array(),
                ));
            }
            // If either side is constant null, the result is constant null.
            (Some(lhs), _) if lhs.is_null() => {
                return Ok(Some(
                    ConstantArray::new(Scalar::null(array.dtype().clone()), array.len())
                        .into_array(),
                ));
            }
            (_, Some(rhs)) if rhs.is_null() => {
                return Ok(Some(
                    ConstantArray::new(Scalar::null(array.dtype().clone()), array.len())
                        .into_array(),
                ));
            }
            _ => {}
        }

        Ok(None)
    }

    fn bind(
        array: &ArithmeticArray,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        // Optimize for constant RHS
        if let Some(rhs_scalar) = array.rhs.as_constant() {
            if rhs_scalar.is_null() {
                // If the RHS is null, the result is always null.
                return ConstantArray::new(Scalar::null(array.dtype().clone()), array.len())
                    .into_array()
                    .bind(selection, ctx);
            }

            let lhs = ctx.bind(&array.lhs, selection)?;
            return match_each_native_ptype!(
                    array.dtype().as_ptype(),
                    integral: |T| {
                        let rhs: T = rhs_scalar
                            .as_primitive()
                            .typed_value::<T>()
                            .vortex_expect("Already checked for null above");
                        Ok(match array.operator() {
                            ArithmeticOperator::Add => checked_arithmetic_scalar_kernel::<Add, T>(lhs, rhs),
                            ArithmeticOperator::Sub => checked_arithmetic_scalar_kernel::<Sub, T>(lhs, rhs),
                            ArithmeticOperator::Mul => checked_arithmetic_scalar_kernel::<Mul, T>(lhs, rhs),
                            ArithmeticOperator::Div => checked_arithmetic_scalar_kernel::<Div, T>(lhs, rhs),
                        })
                    },
                    floating: |T| {
                        let rhs: T = rhs_scalar
                            .as_primitive()
                            .typed_value::<T>()
                            .vortex_expect("Already checked for null above");
                        Ok(match array.operator() {
                            ArithmeticOperator::Add => arithmetic_scalar_kernel::<Add, T>(lhs, rhs),
                            ArithmeticOperator::Sub => arithmetic_scalar_kernel::<Sub, T>(lhs, rhs),
                            ArithmeticOperator::Mul => arithmetic_scalar_kernel::<Mul, T>(lhs, rhs),
                            ArithmeticOperator::Div => arithmetic_scalar_kernel::<Div, T>(lhs, rhs),
                        })
                    }
            );
        }

        let lhs = ctx.bind(&array.lhs, selection)?;
        let rhs = ctx.bind(&array.rhs, selection)?;

        match_each_native_ptype!(
            array.dtype().as_ptype(),
            integral: |T| {
                Ok(match array.operator() {
                    ArithmeticOperator::Add => checked_arithmetic_kernel::<Add, T>(lhs, rhs),
                    ArithmeticOperator::Sub => checked_arithmetic_kernel::<Sub, T>(lhs, rhs),
                    ArithmeticOperator::Mul => checked_arithmetic_kernel::<Mul, T>(lhs, rhs),
                    ArithmeticOperator::Div => checked_arithmetic_kernel::<Div, T>(lhs, rhs),
                })
            },
            floating: |T| {
                Ok(match array.operator() {
                    ArithmeticOperator::Add => arithmetic_kernel::<Add, T>(lhs, rhs),
                    ArithmeticOperator::Sub => arithmetic_kernel::<Sub, T>(lhs, rhs),
                    ArithmeticOperator::Mul => arithmetic_kernel::<Mul, T>(lhs, rhs),
                    ArithmeticOperator::Div => arithmetic_kernel::<Div, T>(lhs, rhs),
                })
            }
        )
    }
}

fn arithmetic_kernel<Op, T>(lhs: BatchKernelRef, rhs: BatchKernelRef) -> BatchKernelRef
where
    T: NativePType,
    Op: Operator<T>,
{
    kernel(move || {
        let lhs = lhs.execute()?.into_primitive().downcast::<T>();
        let rhs = rhs.execute()?.into_primitive().downcast::<T>();
        let result = Arithmetic::<Op, _>::eval(lhs, &rhs);
        Ok(result.into())
    })
}

fn arithmetic_scalar_kernel<Op, T>(lhs: BatchKernelRef, rhs: T) -> BatchKernelRef
where
    T: NativePType + TryFrom<PValue>,
    Op: Operator<T>,
{
    kernel(move || {
        let lhs = lhs.execute()?.into_primitive().downcast::<T>();
        let result = Arithmetic::<Op, _>::eval(lhs, &rhs);
        Ok(result.into())
    })
}

fn checked_arithmetic_kernel<Op, T>(lhs: BatchKernelRef, rhs: BatchKernelRef) -> BatchKernelRef
where
    T: NativePType,
    Op: CheckedOperator<T>,
    PVector<T>: for<'a> CheckedArithmetic<Op, &'a PVector<T>, Output = PVector<T>>,
{
    kernel(move || {
        let lhs = lhs.execute()?.into_primitive().downcast::<T>();
        let rhs = rhs.execute()?.into_primitive().downcast::<T>();
        let result = CheckedArithmetic::<Op, _>::checked_eval(lhs, &rhs)
            .ok_or_else(|| vortex_err!("Arithmetic operation resulted in overflow"))?;
        Ok(result.into())
    })
}

fn checked_arithmetic_scalar_kernel<Op, T>(lhs: BatchKernelRef, rhs: T) -> BatchKernelRef
where
    T: NativePType + TryFrom<PValue>,
    Op: CheckedOperator<T>,
    PVector<T>: for<'a> CheckedArithmetic<Op, &'a T, Output = PVector<T>>,
{
    kernel(move || {
        let lhs = lhs.execute()?.into_primitive().downcast::<T>();
        let result = CheckedArithmetic::<Op, _>::checked_eval(lhs, &rhs)
            .ok_or_else(|| vortex_err!("Arithmetic operation resulted in overflow"))?;
        Ok(result.into())
    })
}

#[cfg(test)]
mod tests {
    use vortex_buffer::{bitbuffer, buffer};
    use vortex_dtype::PTypeDowncastExt;

    use crate::arrays::PrimitiveArray;
    use crate::compute::arrays::arithmetic::{ArithmeticArray, ArithmeticOperator};
    use crate::{ArrayOperator, ArrayRef, IntoArray};

    fn add(lhs: ArrayRef, rhs: ArrayRef) -> ArrayRef {
        ArithmeticArray::new(lhs, rhs, ArithmeticOperator::Add).into_array()
    }

    fn sub(lhs: ArrayRef, rhs: ArrayRef) -> ArrayRef {
        ArithmeticArray::new(lhs, rhs, ArithmeticOperator::Sub).into_array()
    }

    fn mul(lhs: ArrayRef, rhs: ArrayRef) -> ArrayRef {
        ArithmeticArray::new(lhs, rhs, ArithmeticOperator::Mul).into_array()
    }

    fn div(lhs: ArrayRef, rhs: ArrayRef) -> ArrayRef {
        ArithmeticArray::new(lhs, rhs, ArithmeticOperator::Div).into_array()
    }

    #[test]
    fn test_add() {
        let lhs = PrimitiveArray::from_iter([1u32, 2, 3]).into_array();
        let rhs = PrimitiveArray::from_iter([10u32, 20, 30]).into_array();
        let result = add(lhs, rhs)
            .execute()
            .unwrap()
            .into_primitive()
            .downcast::<u32>();
        assert_eq!(result.elements(), &buffer![11u32, 22, 33]);
    }

    #[test]
    fn test_sub() {
        let lhs = PrimitiveArray::from_iter([10u32, 20, 30]).into_array();
        let rhs = PrimitiveArray::from_iter([1u32, 2, 3]).into_array();
        let result = sub(lhs, rhs)
            .execute()
            .unwrap()
            .into_primitive()
            .downcast::<u32>();
        assert_eq!(result.elements(), &buffer![9u32, 18, 27]);
    }

    #[test]
    fn test_mul() {
        let lhs = PrimitiveArray::from_iter([2u32, 3, 4]).into_array();
        let rhs = PrimitiveArray::from_iter([10u32, 20, 30]).into_array();
        let result = mul(lhs, rhs)
            .execute()
            .unwrap()
            .into_primitive()
            .downcast::<u32>();
        assert_eq!(result.elements(), &buffer![20u32, 60, 120]);
    }

    #[test]
    fn test_div() {
        let lhs = PrimitiveArray::from_iter([100u32, 200, 300]).into_array();
        let rhs = PrimitiveArray::from_iter([10u32, 20, 30]).into_array();
        let result = div(lhs, rhs)
            .execute()
            .unwrap()
            .into_primitive()
            .downcast::<u32>();
        assert_eq!(result.elements(), &buffer![10u32, 10, 10]);
    }

    #[test]
    fn test_add_with_selection() {
        let lhs = PrimitiveArray::from_iter([1u32, 2, 3]).into_array();
        let rhs = PrimitiveArray::from_iter([10u32, 20, 30]).into_array();

        let selection = bitbuffer![1 0 1].into_array();

        let result = add(lhs, rhs)
            .execute_with_selection(Some(&selection))
            .unwrap()
            .into_primitive()
            .downcast::<u32>();
        assert_eq!(result.elements(), &buffer![11u32, 33]);
    }
}
