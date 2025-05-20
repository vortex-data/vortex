use core::fmt;
use std::any::Any;
use std::fmt::{Display, Formatter};
use std::sync::LazyLock;

use arcref::ArcRef;
use arrow_buffer::BooleanBuffer;
use arrow_ord::cmp;
use vortex_dtype::{DType, NativePType, Nullability};
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_scalar::Scalar;

use crate::arrays::ConstantArray;
use crate::arrow::{Datum, from_arrow_array_with_len};
use crate::compute::{ComputeFn, ComputeFnVTable, InvocationArgs, Kernel, Options, Output};
use crate::vtable::VTable;
use crate::{Array, ArrayRef, Canonical, IntoArray};

/// Compares two arrays and returns a new boolean array with the result of the comparison.
/// Or, returns None if comparison is not supported for these arrays.
pub fn compare(left: &dyn Array, right: &dyn Array, operator: Operator) -> VortexResult<ArrayRef> {
    COMPARE_FN
        .invoke(&InvocationArgs {
            inputs: &[left.into(), right.into()],
            options: &operator,
        })?
        .unwrap_array()
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd)]
pub enum Operator {
    Eq,
    NotEq,
    Gt,
    Gte,
    Lt,
    Lte,
}

impl Display for Operator {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let display = match &self {
            Operator::Eq => "=",
            Operator::NotEq => "!=",
            Operator::Gt => ">",
            Operator::Gte => ">=",
            Operator::Lt => "<",
            Operator::Lte => "<=",
        };
        Display::fmt(display, f)
    }
}

impl Operator {
    pub fn inverse(self) -> Self {
        match self {
            Operator::Eq => Operator::NotEq,
            Operator::NotEq => Operator::Eq,
            Operator::Gt => Operator::Lte,
            Operator::Gte => Operator::Lt,
            Operator::Lt => Operator::Gte,
            Operator::Lte => Operator::Gt,
        }
    }

    /// Change the sides of the operator, where changing lhs and rhs won't change the result of the operation
    pub fn swap(self) -> Self {
        match self {
            Operator::Eq => Operator::Eq,
            Operator::NotEq => Operator::NotEq,
            Operator::Gt => Operator::Lt,
            Operator::Gte => Operator::Lte,
            Operator::Lt => Operator::Gt,
            Operator::Lte => Operator::Gte,
        }
    }
}

pub struct CompareKernelRef(ArcRef<dyn Kernel>);
inventory::collect!(CompareKernelRef);

pub trait CompareKernel: VTable {
    fn compare(
        &self,
        lhs: &Self::Array,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>>;
}

#[derive(Debug)]
pub struct CompareKernelAdapter<V: VTable>(pub V);

impl<V: VTable + CompareKernel> CompareKernelAdapter<V> {
    pub const fn lift(&'static self) -> CompareKernelRef {
        CompareKernelRef(ArcRef::new_ref(self))
    }
}

impl<V: VTable + CompareKernel> Kernel for CompareKernelAdapter<V> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let inputs = CompareArgs::try_from(args)?;
        let Some(array) = inputs.lhs.as_opt::<V>() else {
            return Ok(None);
        };
        Ok(V::compare(&self.0, array, inputs.rhs, inputs.operator)?.map(|array| array.into()))
    }
}

pub static COMPARE_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("compare".into(), ArcRef::new_ref(&Compare));
    for kernel in inventory::iter::<CompareKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

struct Compare;

impl ComputeFnVTable for Compare {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        let CompareArgs { lhs, rhs, operator } = CompareArgs::try_from(args)?;

        let return_dtype = self.return_dtype(args)?;

        if lhs.is_empty() {
            return Ok(Canonical::empty(&return_dtype).into_array().into());
        }

        let left_constant_null = lhs.as_constant().map(|l| l.is_null()).unwrap_or(false);
        let right_constant_null = rhs.as_constant().map(|r| r.is_null()).unwrap_or(false);
        if left_constant_null || right_constant_null {
            return Ok(ConstantArray::new(Scalar::null(return_dtype), lhs.len())
                .into_array()
                .into());
        }

        let right_is_constant = rhs.is_constant();

        // Always try to put constants on the right-hand side so encodings can optimise themselves.
        if lhs.is_constant() && !right_is_constant {
            return Ok(compare(rhs, lhs, operator.swap())?.into());
        }

        // First try lhs op rhs, then invert and try again.
        for kernel in kernels {
            if let Some(output) = kernel.invoke(args)? {
                return Ok(output);
            }
        }
        if let Some(output) = lhs.invoke(&COMPARE_FN, args)? {
            return Ok(output);
        }

        // Try inverting the operator and swapping the arguments
        let inverted_args = InvocationArgs {
            inputs: &[rhs.into(), lhs.into()],
            options: &operator.swap(),
        };
        for kernel in kernels {
            if let Some(output) = kernel.invoke(&inverted_args)? {
                return Ok(output);
            }
        }
        if let Some(output) = rhs.invoke(&COMPARE_FN, &inverted_args)? {
            return Ok(output);
        }

        // Only log missing compare implementation if there's possibly better one than arrow,
        // i.e. lhs isn't arrow or rhs isn't arrow or constant
        if !(lhs.is_arrow() && (rhs.is_arrow() || right_is_constant)) {
            log::debug!(
                "No compare implementation found for LHS {}, RHS {}, and operator {} (or inverse)",
                rhs.encoding_id(),
                lhs.encoding_id(),
                operator.swap(),
            );
        }

        // Fallback to arrow on canonical types
        Ok(arrow_compare(lhs, rhs, operator)?.into())
    }

    fn return_dtype(&self, args: &InvocationArgs) -> VortexResult<DType> {
        let CompareArgs { lhs, rhs, .. } = CompareArgs::try_from(args)?;

        if !lhs.dtype().eq_ignore_nullability(rhs.dtype()) {
            vortex_bail!(
                "Cannot compare different DTypes {} and {}",
                lhs.dtype(),
                rhs.dtype()
            );
        }

        // TODO(ngates): no reason why not
        if lhs.dtype().is_struct() {
            vortex_bail!(
                "Compare does not support arrays with Struct DType, got: {} and {}",
                lhs.dtype(),
                rhs.dtype()
            )
        }

        Ok(DType::Bool(
            (lhs.dtype().is_nullable() || rhs.dtype().is_nullable()).into(),
        ))
    }

    fn return_len(&self, args: &InvocationArgs) -> VortexResult<usize> {
        let CompareArgs { lhs, rhs, .. } = CompareArgs::try_from(args)?;
        if lhs.len() != rhs.len() {
            vortex_bail!(
                "Compare operations only support arrays of the same length, got {} and {}",
                lhs.len(),
                rhs.len()
            );
        }
        Ok(lhs.len())
    }

    fn is_elementwise(&self) -> bool {
        true
    }
}

struct CompareArgs<'a> {
    lhs: &'a dyn Array,
    rhs: &'a dyn Array,
    operator: Operator,
}

impl Options for Operator {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl<'a> TryFrom<&InvocationArgs<'a>> for CompareArgs<'a> {
    type Error = VortexError;

    fn try_from(value: &InvocationArgs<'a>) -> Result<Self, Self::Error> {
        if value.inputs.len() != 2 {
            vortex_bail!("Expected 2 inputs, found {}", value.inputs.len());
        }
        let lhs = value.inputs[0]
            .array()
            .ok_or_else(|| vortex_err!("Expected first input to be an array"))?;
        let rhs = value.inputs[1]
            .array()
            .ok_or_else(|| vortex_err!("Expected second input to be an array"))?;
        let operator = *value
            .options
            .as_any()
            .downcast_ref::<Operator>()
            .vortex_expect("Expected options to be an operator");

        Ok(CompareArgs { lhs, rhs, operator })
    }
}

/// Helper function to compare empty values with arrays that have external value length information
/// like `VarBin`.
pub fn compare_lengths_to_empty<P, I>(lengths: I, op: Operator) -> BooleanBuffer
where
    P: NativePType,
    I: Iterator<Item = P>,
{
    // All comparison can be expressed in terms of equality. "" is the absolute min of possible value.
    let cmp_fn = match op {
        Operator::Eq | Operator::Lte => |v| v == P::zero(),
        Operator::NotEq | Operator::Gt => |v| v != P::zero(),
        Operator::Gte => |_| true,
        Operator::Lt => |_| false,
    };

    lengths.map(cmp_fn).collect::<BooleanBuffer>()
}

/// Implementation of `CompareFn` using the Arrow crate.
fn arrow_compare(
    left: &dyn Array,
    right: &dyn Array,
    operator: Operator,
) -> VortexResult<ArrayRef> {
    let nullable = left.dtype().is_nullable() || right.dtype().is_nullable();
    let lhs = Datum::try_new(left)?;
    let rhs = Datum::try_new(right)?;

    let array = match operator {
        Operator::Eq => cmp::eq(&lhs, &rhs)?,
        Operator::NotEq => cmp::neq(&lhs, &rhs)?,
        Operator::Gt => cmp::gt(&lhs, &rhs)?,
        Operator::Gte => cmp::gt_eq(&lhs, &rhs)?,
        Operator::Lt => cmp::lt(&lhs, &rhs)?,
        Operator::Lte => cmp::lt_eq(&lhs, &rhs)?,
    };
    from_arrow_array_with_len(&array, left.len(), nullable)
}

pub fn scalar_cmp(lhs: &Scalar, rhs: &Scalar, operator: Operator) -> Scalar {
    if lhs.is_null() | rhs.is_null() {
        Scalar::null(DType::Bool(Nullability::Nullable))
    } else {
        let b = match operator {
            Operator::Eq => lhs == rhs,
            Operator::NotEq => lhs != rhs,
            Operator::Gt => lhs > rhs,
            Operator::Gte => lhs >= rhs,
            Operator::Lt => lhs < rhs,
            Operator::Lte => lhs <= rhs,
        };

        Scalar::bool(
            b,
            (lhs.dtype().is_nullable() || rhs.dtype().is_nullable()).into(),
        )
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::BooleanBuffer;
    use rstest::rstest;

    use super::*;
    use crate::ToCanonical;
    use crate::arrays::{BoolArray, ConstantArray, VarBinArray, VarBinViewArray};
    use crate::test_harness::to_int_indices;
    use crate::validity::Validity;

    #[test]
    fn test_bool_basic_comparisons() {
        let arr = BoolArray::new(
            BooleanBuffer::from_iter([true, true, false, true, false]),
            Validity::from_iter([false, true, true, true, true]),
        );

        let matches = compare(arr.as_ref(), arr.as_ref(), Operator::Eq)
            .unwrap()
            .to_bool()
            .unwrap();

        assert_eq!(to_int_indices(matches).unwrap(), [1u64, 2, 3, 4]);

        let matches = compare(arr.as_ref(), arr.as_ref(), Operator::NotEq)
            .unwrap()
            .to_bool()
            .unwrap();
        let empty: [u64; 0] = [];
        assert_eq!(to_int_indices(matches).unwrap(), empty);

        let other = BoolArray::new(
            BooleanBuffer::from_iter([false, false, false, true, true]),
            Validity::from_iter([false, true, true, true, true]),
        );

        let matches = compare(arr.as_ref(), other.as_ref(), Operator::Lte)
            .unwrap()
            .to_bool()
            .unwrap();
        assert_eq!(to_int_indices(matches).unwrap(), [2u64, 3, 4]);

        let matches = compare(arr.as_ref(), other.as_ref(), Operator::Lt)
            .unwrap()
            .to_bool()
            .unwrap();
        assert_eq!(to_int_indices(matches).unwrap(), [4u64]);

        let matches = compare(other.as_ref(), arr.as_ref(), Operator::Gte)
            .unwrap()
            .to_bool()
            .unwrap();
        assert_eq!(to_int_indices(matches).unwrap(), [2u64, 3, 4]);

        let matches = compare(other.as_ref(), arr.as_ref(), Operator::Gt)
            .unwrap()
            .to_bool()
            .unwrap();
        assert_eq!(to_int_indices(matches).unwrap(), [4u64]);
    }

    #[test]
    fn constant_compare() {
        let left = ConstantArray::new(Scalar::from(2u32), 10);
        let right = ConstantArray::new(Scalar::from(10u32), 10);

        let compare = compare(left.as_ref(), right.as_ref(), Operator::Gt).unwrap();
        let res = compare.as_constant().unwrap();
        assert_eq!(res.as_bool().value(), Some(false));
        assert_eq!(compare.len(), 10);

        let compare = arrow_compare(&left.into_array(), &right.into_array(), Operator::Gt).unwrap();
        let res = compare.as_constant().unwrap();
        assert_eq!(res.as_bool().value(), Some(false));
        assert_eq!(compare.len(), 10);
    }

    #[rstest]
    #[case(Operator::Eq, vec![false, false, false, true])]
    #[case(Operator::NotEq, vec![true, true, true, false])]
    #[case(Operator::Gt, vec![true, true, true, false])]
    #[case(Operator::Gte, vec![true, true, true, true])]
    #[case(Operator::Lt, vec![false, false, false, false])]
    #[case(Operator::Lte, vec![false, false, false, true])]
    fn test_cmp_to_empty(#[case] op: Operator, #[case] expected: Vec<bool>) {
        let lengths: Vec<i32> = vec![1, 5, 7, 0];

        let output = compare_lengths_to_empty(lengths.iter().copied(), op);
        assert_eq!(Vec::from_iter(output.iter()), expected);
    }

    #[rstest]
    #[case(VarBinArray::from(vec!["a", "b"]).into_array(), VarBinViewArray::from_iter_str(["a", "b"]).into_array())]
    #[case(VarBinViewArray::from_iter_str(["a", "b"]).into_array(), VarBinArray::from(vec!["a", "b"]).into_array())]
    #[case(VarBinArray::from(vec!["a".as_bytes(), "b".as_bytes()]).into_array(), VarBinViewArray::from_iter_bin(["a".as_bytes(), "b".as_bytes()]).into_array())]
    #[case(VarBinViewArray::from_iter_bin(["a".as_bytes(), "b".as_bytes()]).into_array(), VarBinArray::from(vec!["a".as_bytes(), "b".as_bytes()]).into_array())]
    fn arrow_compare_different_encodings(#[case] left: ArrayRef, #[case] right: ArrayRef) {
        let res = compare(&left, &right, Operator::Eq).unwrap();
        assert_eq!(
            res.to_bool().unwrap().boolean_buffer().count_set_bits(),
            left.len()
        );
    }
}
