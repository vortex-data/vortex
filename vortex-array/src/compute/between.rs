use std::any::Any;
use std::sync::LazyLock;

use arcref::ArcRef;
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_scalar::Scalar;

use crate::arrays::ConstantArray;
use crate::compute::{
    BooleanOperator, ComputeFn, ComputeFnVTable, InvocationArgs, Kernel, Operator, Options, Output,
    boolean, compare,
};
use crate::vtable::VTable;
use crate::{Array, ArrayRef, Canonical, IntoArray};

/// Compute between (a <= x <= b), this can be implemented using compare and boolean and but this
/// will likely have a lower runtime.
///
/// This semantics is equivalent to:
/// ```
/// use vortex_array::{Array, ArrayRef};
/// use vortex_array::compute::{boolean, compare, BetweenOptions, BooleanOperator, Operator};///
/// use vortex_error::VortexResult;
///
/// fn between(
///    arr: &dyn Array,
///    lower: &dyn Array,
///    upper: &dyn Array,
///    options: &BetweenOptions
/// ) -> VortexResult<ArrayRef> {
///     boolean(
///         &compare(lower, arr, options.lower_strict.to_operator())?,
///         &compare(arr, upper,  options.upper_strict.to_operator())?,
///         BooleanOperator::And
///     )
/// }
///  ```
///
/// The BetweenOptions { lower: StrictComparison, upper: StrictComparison } defines if the
/// value is < (strict) or <= (non-strict).
///
pub fn between(
    arr: &dyn Array,
    lower: &dyn Array,
    upper: &dyn Array,
    options: &BetweenOptions,
) -> VortexResult<ArrayRef> {
    BETWEEN_FN
        .invoke(&InvocationArgs {
            inputs: &[arr.into(), lower.into(), upper.into()],
            options,
        })?
        .unwrap_array()
}

pub struct BetweenKernelRef(ArcRef<dyn Kernel>);
inventory::collect!(BetweenKernelRef);

pub trait BetweenKernel: VTable {
    fn between(
        &self,
        arr: &Self::Array,
        lower: &dyn Array,
        upper: &dyn Array,
        options: &BetweenOptions,
    ) -> VortexResult<Option<ArrayRef>>;
}

#[derive(Debug)]
pub struct BetweenKernelAdapter<V: VTable>(pub V);

impl<V: VTable + BetweenKernel> BetweenKernelAdapter<V> {
    pub const fn lift(&'static self) -> BetweenKernelRef {
        BetweenKernelRef(ArcRef::new_ref(self))
    }
}

impl<V: VTable + BetweenKernel> Kernel for BetweenKernelAdapter<V> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let inputs = BetweenArgs::try_from(args)?;
        let Some(array) = inputs.array.as_opt::<V>() else {
            return Ok(None);
        };
        Ok(
            V::between(&self.0, array, inputs.lower, inputs.upper, inputs.options)?
                .map(|array| array.into()),
        )
    }
}

pub static BETWEEN_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("between".into(), ArcRef::new_ref(&Between));
    for kernel in inventory::iter::<BetweenKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

struct Between;

impl ComputeFnVTable for Between {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        let BetweenArgs {
            array,
            lower,
            upper,
            options,
        } = BetweenArgs::try_from(args)?;

        let return_dtype = self.return_dtype(args)?;

        // Bail early if the array is empty.
        if array.is_empty() {
            return Ok(Canonical::empty(&return_dtype).into_array().into());
        }

        // A quick check to see if either array might is a null constant array.
        // Note: Depends on returning early if array is empty for is_invalid check.
        if lower.is_invalid(0)? || upper.is_invalid(0)? {
            if let (Some(c_lower), Some(c_upper)) = (lower.as_constant(), upper.as_constant()) {
                if c_lower.is_null() || c_upper.is_null() {
                    return Ok(ConstantArray::new(Scalar::null(return_dtype), array.len())
                        .into_array()
                        .into());
                }
            }
        }

        if lower.as_constant().is_some_and(|v| v.is_null())
            || upper.as_constant().is_some_and(|v| v.is_null())
        {
            return Ok(ConstantArray::new(Scalar::null(return_dtype), array.len())
                .into_array()
                .into());
        }

        // Try each kernel
        for kernel in kernels {
            if let Some(output) = kernel.invoke(args)? {
                return Ok(output);
            }
        }
        if let Some(output) = array.invoke(&BETWEEN_FN, args)? {
            return Ok(output);
        }

        // Otherwise, fall back to the default Arrow implementation
        // TODO(joe): should we try to canonicalize the array and try between
        Ok(boolean(
            &compare(lower, array, options.lower_strict.to_operator())?,
            &compare(array, upper, options.upper_strict.to_operator())?,
            BooleanOperator::And,
        )?
        .into())
    }

    fn return_dtype(&self, args: &InvocationArgs) -> VortexResult<DType> {
        let BetweenArgs {
            array,
            lower,
            upper,
            options: _,
        } = BetweenArgs::try_from(args)?;

        if !array.dtype().eq_ignore_nullability(lower.dtype()) {
            vortex_bail!(
                "Array and lower bound types do not match: {:?} != {:?}",
                array.dtype(),
                lower.dtype()
            );
        }
        if !array.dtype().eq_ignore_nullability(upper.dtype()) {
            vortex_bail!(
                "Array and upper bound types do not match: {:?} != {:?}",
                array.dtype(),
                upper.dtype()
            );
        }

        Ok(DType::Bool(
            array.dtype().nullability() | lower.dtype().nullability() | upper.dtype().nullability(),
        ))
    }

    fn return_len(&self, args: &InvocationArgs) -> VortexResult<usize> {
        let BetweenArgs {
            array,
            lower,
            upper,
            options: _,
        } = BetweenArgs::try_from(args)?;
        if array.len() != lower.len() || array.len() != upper.len() {
            vortex_bail!(
                "Array lengths do not match: array:{} lower:{} upper:{}",
                array.len(),
                lower.len(),
                upper.len()
            );
        }
        Ok(array.len())
    }

    fn is_elementwise(&self) -> bool {
        true
    }
}

struct BetweenArgs<'a> {
    array: &'a dyn Array,
    lower: &'a dyn Array,
    upper: &'a dyn Array,
    options: &'a BetweenOptions,
}

impl<'a> TryFrom<&InvocationArgs<'a>> for BetweenArgs<'a> {
    type Error = VortexError;

    fn try_from(value: &InvocationArgs<'a>) -> VortexResult<Self> {
        if value.inputs.len() != 3 {
            vortex_bail!("Expected 3 inputs, found {}", value.inputs.len());
        }
        let array = value.inputs[0]
            .array()
            .ok_or_else(|| vortex_err!("Expected input 0 to be an array"))?;
        let lower = value.inputs[1]
            .array()
            .ok_or_else(|| vortex_err!("Expected input 1 to be an array"))?;
        let upper = value.inputs[2]
            .array()
            .ok_or_else(|| vortex_err!("Expected input 2 to be an array"))?;
        let options = value
            .options
            .as_any()
            .downcast_ref::<BetweenOptions>()
            .vortex_expect("Expected options to be an operator");

        Ok(BetweenArgs {
            array,
            lower,
            upper,
            options,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BetweenOptions {
    pub lower_strict: StrictComparison,
    pub upper_strict: StrictComparison,
}

impl Options for BetweenOptions {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum StrictComparison {
    Strict,
    NonStrict,
}

impl StrictComparison {
    pub const fn to_operator(&self) -> Operator {
        match self {
            StrictComparison::Strict => Operator::Lt,
            StrictComparison::NonStrict => Operator::Lte,
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::{Nullability, PType};

    use super::*;
    use crate::ToCanonical;
    use crate::arrays::PrimitiveArray;
    use crate::compute::conformance::search_sorted::rstest;
    use crate::test_harness::to_int_indices;

    #[rstest]
    #[case(StrictComparison::NonStrict, StrictComparison::NonStrict, vec![0, 1, 2, 3])]
    #[case(StrictComparison::NonStrict, StrictComparison::Strict, vec![0, 1])]
    #[case(StrictComparison::Strict, StrictComparison::NonStrict, vec![0, 2])]
    #[case(StrictComparison::Strict, StrictComparison::Strict, vec![0])]
    fn test_bounds(
        #[case] lower_strict: StrictComparison,
        #[case] upper_strict: StrictComparison,
        #[case] expected: Vec<u64>,
    ) {
        let lower = PrimitiveArray::from_iter([0, 0, 0, 0, 2]);
        let array = PrimitiveArray::from_iter([1, 0, 1, 0, 1]);
        let upper = PrimitiveArray::from_iter([2, 1, 1, 0, 0]);

        let matches = between(
            array.as_ref(),
            lower.as_ref(),
            upper.as_ref(),
            &BetweenOptions {
                lower_strict,
                upper_strict,
            },
        )
        .unwrap()
        .to_bool()
        .unwrap();

        let indices = to_int_indices(matches).unwrap();
        assert_eq!(indices, expected);
    }

    #[test]
    fn test_constants() {
        let lower = PrimitiveArray::from_iter([0, 0, 2, 0, 2]);
        let array = PrimitiveArray::from_iter([1, 0, 1, 0, 1]);

        // upper is null
        let upper = ConstantArray::new(
            Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
            5,
        );

        let matches = between(
            array.as_ref(),
            lower.as_ref(),
            upper.as_ref(),
            &BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        )
        .unwrap()
        .to_bool()
        .unwrap();

        let indices = to_int_indices(matches).unwrap();
        assert!(indices.is_empty());

        // upper is a fixed constant
        let upper = ConstantArray::new(Scalar::from(2), 5);
        let matches = between(
            array.as_ref(),
            lower.as_ref(),
            upper.as_ref(),
            &BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        )
        .unwrap()
        .to_bool()
        .unwrap();
        let indices = to_int_indices(matches).unwrap();
        assert_eq!(indices, vec![0, 1, 3]);

        // lower is also a constant
        let lower = ConstantArray::new(Scalar::from(0), 5);

        let matches = between(
            array.as_ref(),
            lower.as_ref(),
            upper.as_ref(),
            &BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        )
        .unwrap()
        .to_bool()
        .unwrap();
        let indices = to_int_indices(matches).unwrap();
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }
}
