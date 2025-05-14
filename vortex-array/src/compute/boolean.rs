use std::any::Any;
use std::sync::{Arc, LazyLock};

use arcref::ArcRef;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};

use crate::arrow::{FromArrowArray, IntoArrowArray};
use crate::compute::{ComputeFn, ComputeFnVTable, InvocationArgs, Kernel, Options, Output};
use crate::vtable::VTable;
use crate::{Array, ArrayRef};

/// Point-wise logical _and_ between two Boolean arrays.
///
/// This method uses Arrow-style null propagation rather than the Kleene logic semantics.
pub fn and(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    boolean(lhs, rhs, BooleanOperator::And)
}

/// Point-wise Kleene logical _and_ between two Boolean arrays.
pub fn and_kleene(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    boolean(lhs, rhs, BooleanOperator::AndKleene)
}

/// Point-wise logical _or_ between two Boolean arrays.
///
/// This method uses Arrow-style null propagation rather than the Kleene logic semantics.
pub fn or(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    boolean(lhs, rhs, BooleanOperator::Or)
}

/// Point-wise Kleene logical _or_ between two Boolean arrays.
pub fn or_kleene(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    boolean(lhs, rhs, BooleanOperator::OrKleene)
}

/// Point-wise logical operator between two Boolean arrays.
///
/// This method uses Arrow-style null propagation rather than the Kleene logic semantics.
pub fn boolean(lhs: &dyn Array, rhs: &dyn Array, op: BooleanOperator) -> VortexResult<ArrayRef> {
    BOOLEAN_FN
        .invoke(&InvocationArgs {
            inputs: &[lhs.into(), rhs.into()],
            options: &op,
        })?
        .unwrap_array()
}

pub struct BooleanKernelRef(ArcRef<dyn Kernel>);
inventory::collect!(BooleanKernelRef);

pub trait BooleanKernel: VTable {
    fn boolean(
        &self,
        array: &Self::Array,
        other: &dyn Array,
        op: BooleanOperator,
    ) -> VortexResult<Option<ArrayRef>>;
}

#[derive(Debug)]
pub struct BooleanKernelAdapter<V: VTable>(pub V);

impl<V: VTable + BooleanKernel> BooleanKernelAdapter<V> {
    pub const fn lift(&'static self) -> BooleanKernelRef {
        BooleanKernelRef(ArcRef::new_ref(self))
    }
}

impl<V: VTable + BooleanKernel> Kernel for BooleanKernelAdapter<V> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let inputs = BooleanArgs::try_from(args)?;
        let Some(array) = inputs.lhs.as_opt::<V>() else {
            return Ok(None);
        };
        Ok(V::boolean(&self.0, array, inputs.rhs, inputs.operator)?.map(|array| array.into()))
    }
}

pub static BOOLEAN_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("boolean".into(), ArcRef::new_ref(&Boolean));
    for kernel in inventory::iter::<BooleanKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

struct Boolean;

impl ComputeFnVTable for Boolean {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        let BooleanArgs { lhs, rhs, operator } = BooleanArgs::try_from(args)?;

        let rhs_is_constant = rhs.is_constant();

        // If LHS is constant, then we make sure it's on the RHS.
        if lhs.is_constant() && !rhs_is_constant {
            return Ok(boolean(rhs, lhs, operator)?.into());
        }

        // If the RHS is constant and the LHS is Arrow, we can't do any better than arrow_compare.
        if lhs.is_arrow() && (rhs.is_arrow() || rhs_is_constant) {
            return Ok(arrow_boolean(lhs.to_array(), rhs.to_array(), operator)?.into());
        }

        // Check if either LHS or RHS supports the operation directly.
        for kernel in kernels {
            if let Some(output) = kernel.invoke(args)? {
                return Ok(output);
            }
        }
        if let Some(output) = lhs.invoke(&BOOLEAN_FN, args)? {
            return Ok(output);
        }

        let inverse_args = InvocationArgs {
            inputs: &[rhs.into(), lhs.into()],
            options: &operator,
        };
        for kernel in kernels {
            if let Some(output) = kernel.invoke(&inverse_args)? {
                return Ok(output);
            }
        }
        if let Some(output) = rhs.invoke(&BOOLEAN_FN, &inverse_args)? {
            return Ok(output);
        }

        log::debug!(
            "No boolean implementation found for LHS {}, RHS {}, and operator {:?} (or inverse)",
            rhs.encoding_id(),
            lhs.encoding_id(),
            operator,
        );

        // If neither side implements the trait, then we delegate to Arrow compute.
        Ok(arrow_boolean(lhs.to_array(), rhs.to_array(), operator)?.into())
    }

    fn return_dtype(&self, args: &InvocationArgs) -> VortexResult<DType> {
        let BooleanArgs { lhs, rhs, .. } = BooleanArgs::try_from(args)?;

        if !lhs.dtype().is_boolean()
            || !rhs.dtype().is_boolean()
            || !lhs.dtype().eq_ignore_nullability(rhs.dtype())
        {
            vortex_bail!(
                "Boolean operations are only supported on boolean arrays: {} and {}",
                lhs.dtype(),
                rhs.dtype()
            )
        }

        Ok(DType::Bool(
            (lhs.dtype().is_nullable() || rhs.dtype().is_nullable()).into(),
        ))
    }

    fn return_len(&self, args: &InvocationArgs) -> VortexResult<usize> {
        let BooleanArgs { lhs, rhs, .. } = BooleanArgs::try_from(args)?;

        if lhs.len() != rhs.len() {
            vortex_bail!(
                "Boolean operations aren't supported on arrays of different lengths: {} and {}",
                lhs.len(),
                rhs.len()
            )
        }

        Ok(lhs.len())
    }

    fn is_elementwise(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOperator {
    And,
    AndKleene,
    Or,
    OrKleene,
    // AndNot,
    // AndNotKleene,
    // Xor,
}

impl Options for BooleanOperator {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct BooleanArgs<'a> {
    lhs: &'a dyn Array,
    rhs: &'a dyn Array,
    operator: BooleanOperator,
}

impl<'a> TryFrom<&InvocationArgs<'a>> for BooleanArgs<'a> {
    type Error = VortexError;

    fn try_from(value: &InvocationArgs<'a>) -> VortexResult<Self> {
        if value.inputs.len() != 2 {
            vortex_bail!("Expected 2 inputs, found {}", value.inputs.len());
        }
        let lhs = value.inputs[0]
            .array()
            .ok_or_else(|| vortex_err!("Expected input 0 to be an array"))?;
        let rhs = value.inputs[1]
            .array()
            .ok_or_else(|| vortex_err!("Expected input 1 to be an array"))?;
        let operator = value
            .options
            .as_any()
            .downcast_ref::<BooleanOperator>()
            .vortex_expect("Expected options to be an operator");

        Ok(BooleanArgs {
            lhs,
            rhs,
            operator: *operator,
        })
    }
}

/// Implementation of `BinaryBooleanFn` using the Arrow crate.
///
/// Note that other encodings should handle a constant RHS value, so we can assume here that
/// the RHS is not constant and expand to a full array.
pub(crate) fn arrow_boolean(
    lhs: ArrayRef,
    rhs: ArrayRef,
    operator: BooleanOperator,
) -> VortexResult<ArrayRef> {
    let nullable = lhs.dtype().is_nullable() || rhs.dtype().is_nullable();

    let lhs = lhs.into_arrow(&DataType::Boolean)?.as_boolean().clone();
    let rhs = rhs.into_arrow(&DataType::Boolean)?.as_boolean().clone();

    let array = match operator {
        BooleanOperator::And => arrow_arith::boolean::and(&lhs, &rhs)?,
        BooleanOperator::AndKleene => arrow_arith::boolean::and_kleene(&lhs, &rhs)?,
        BooleanOperator::Or => arrow_arith::boolean::or(&lhs, &rhs)?,
        BooleanOperator::OrKleene => arrow_arith::boolean::or_kleene(&lhs, &rhs)?,
    };

    Ok(ArrayRef::from_arrow(
        Arc::new(array) as ArrowArrayRef,
        nullable,
    ))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::canonical::ToCanonical;
    #[rstest]
    #[case(BoolArray::from_iter([Some(true), Some(true), Some(false), Some(false)].into_iter())
    .into_array(), BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter())
    .into_array())]
    #[case(BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter()).into_array(),
        BoolArray::from_iter([Some(true), Some(true), Some(false), Some(false)].into_iter()).into_array())]
    fn test_or(#[case] lhs: ArrayRef, #[case] rhs: ArrayRef) {
        let r = or(&lhs, &rhs).unwrap();

        let r = r.to_bool().unwrap().into_array();

        let v0 = r.scalar_at(0).unwrap().as_bool().value();
        let v1 = r.scalar_at(1).unwrap().as_bool().value();
        let v2 = r.scalar_at(2).unwrap().as_bool().value();
        let v3 = r.scalar_at(3).unwrap().as_bool().value();

        assert!(v0.unwrap());
        assert!(v1.unwrap());
        assert!(v2.unwrap());
        assert!(!v3.unwrap());
    }

    #[rstest]
    #[case(BoolArray::from_iter([Some(true), Some(true), Some(false), Some(false)].into_iter())
    .into_array(), BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter())
    .into_array())]
    #[case(BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter()).into_array(),
        BoolArray::from_iter([Some(true), Some(true), Some(false), Some(false)].into_iter()).into_array())]
    fn test_and(#[case] lhs: ArrayRef, #[case] rhs: ArrayRef) {
        let r = and(&lhs, &rhs).unwrap().to_bool().unwrap().into_array();

        let v0 = r.scalar_at(0).unwrap().as_bool().value();
        let v1 = r.scalar_at(1).unwrap().as_bool().value();
        let v2 = r.scalar_at(2).unwrap().as_bool().value();
        let v3 = r.scalar_at(3).unwrap().as_bool().value();

        assert!(v0.unwrap());
        assert!(!v1.unwrap());
        assert!(!v2.unwrap());
        assert!(!v3.unwrap());
    }
}
