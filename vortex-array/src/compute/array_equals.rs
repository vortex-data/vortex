// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::LazyLock;

use arcref::ArcRef;
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_scalar::Scalar;

use crate::Array;
use crate::compute::{
    ComputeFn, ComputeFnVTable, InvocationArgs, Kernel, Operator, Options, Output, compare,
};
use crate::stats::{Precision, Stat, StatsProvider};
use crate::vtable::VTable;

pub fn array_equals(left: &dyn Array, right: &dyn Array) -> VortexResult<bool> {
    array_equals_opts(left, right, false)
}

pub fn array_equals_opts(
    left: &dyn Array,
    right: &dyn Array,
    ignore_nullability: bool,
) -> VortexResult<bool> {
    Ok(ARRAY_EQUALS_FN
        .invoke(&InvocationArgs {
            inputs: &[left.into(), right.into()],
            options: &ArrayEqualsOptions { ignore_nullability },
        })?
        .unwrap_scalar()?
        .as_bool()
        .value()
        .vortex_expect("non-nullable"))
}

#[derive(Clone, Copy)]
struct ArrayEqualsOptions {
    ignore_nullability: bool,
}

impl Options for ArrayEqualsOptions {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub static ARRAY_EQUALS_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("array_equals".into(), ArcRef::new_ref(&ArrayEquals));
    for kernel in inventory::iter::<ArrayEqualsKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

struct ArrayEquals;
impl ComputeFnVTable for ArrayEquals {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        let ArrayEqualsArgs {
            left,
            right,
            ignore_nullability,
        } = ArrayEqualsArgs::try_from(args)?;

        if ignore_nullability && !left.dtype().eq_ignore_nullability(right.dtype()) {
            return Ok(Scalar::from(false).into());
        }

        if !ignore_nullability && !left.dtype().eq(right.dtype()) {
            return Ok(Scalar::from(false).into());
        }

        if left.len() != right.len() {
            return Ok(Scalar::from(false).into());
        }

        if let Some(l_scalar) = left.as_constant()
            && let Some(r_scalar) = right.as_constant()
        {
            return Ok(Scalar::from(l_scalar.eq(&r_scalar)).into());
        }

        if left.is_empty() && right.is_empty() {
            return Ok(Scalar::from(true).into());
        }

        for stat in [
            Stat::IsConstant,
            Stat::IsSorted,
            Stat::IsStrictSorted,
            Stat::Max, // todo: can we do that with e.g. float errors?
            Stat::Min,
            Stat::Sum,
            Stat::NullCount,
            Stat::NaNCount,
            // No Stat::UncompressedSizeInBytes because arrays may physically differ and has a different metric
        ] {
            let Some(Precision::Exact(left_v)) = left.statistics().get(stat) else {
                continue;
            };

            let Some(Precision::Exact(right_v)) = right.statistics().get(stat) else {
                continue;
            };

            if !left_v.eq(&right_v) {
                return Ok(Scalar::from(false).into());
            }
        }

        let args = InvocationArgs {
            inputs: &[left.into(), right.into()],
            options: &ArrayEqualsOptions { ignore_nullability },
        };

        for kernel in kernels {
            if let Some(output) = kernel.invoke(&args)? {
                return Ok(output);
            }
        }

        if let Some(output) = left.invoke(&ARRAY_EQUALS_FN, &args)? {
            return Ok(output);
        }

        // Try swapping arguments
        let swapped_args = InvocationArgs {
            inputs: &[right.into(), left.into()],
            options: &ArrayEqualsOptions { ignore_nullability },
        };
        if let Some(output) = right.invoke(&ARRAY_EQUALS_FN, &swapped_args)? {
            return Ok(output);
        }

        // Try canonical arrays if not already canonical
        let canonical_equals = if !left.is_canonical() || !right.is_canonical() {
            let left_canonical = left.to_canonical()?;
            let right_canonical = right.to_canonical()?;

            array_equals_opts(
                left_canonical.as_ref(),
                right_canonical.as_ref(),
                ignore_nullability,
            )?
        } else {
            // Fallback to chunked comparison
            const BATCH_SIZE: usize = 65536; // 64K elements per batch

            let mut offset = 0;
            while offset < left.len() {
                let end = (offset + BATCH_SIZE).min(left.len());

                let left_slice = left.slice(offset, end)?;
                let right_slice = right.slice(offset, end)?;

                let compare_result = compare(&left_slice, &right_slice, Operator::Eq)?;

                // For array equality, we need to check if all values are equal
                // This includes treating NULL == NULL as true
                let all_equal = if let Some(constant_scalar) = compare_result.as_constant() {
                    // If constant is true, all are equal
                    constant_scalar.is_valid() && constant_scalar.as_bool().value() == Some(true)
                } else {
                    // Not constant - need to check each value
                    let mut found_inequality = false;
                    for i in 0..compare_result.len() {
                        let cmp_scalar = compare_result.scalar_at(i)?;
                        if cmp_scalar.is_valid() && cmp_scalar.as_bool().value() == Some(false) {
                            // Found a definite inequality
                            found_inequality = true;
                            break;
                        }
                        // For null comparison results, we need to check the original values
                        if cmp_scalar.is_null() {
                            let left_val = left_slice.scalar_at(i)?;
                            let right_val = right_slice.scalar_at(i)?;
                            // If both are null, they're equal; if only one is null, they're not
                            if left_val.is_null() != right_val.is_null() {
                                found_inequality = true;
                                break;
                            }
                        }
                    }
                    !found_inequality
                };

                if !all_equal {
                    return Ok(Scalar::from(false).into());
                }

                offset = end;
            }

            true
        };

        Ok(Scalar::from(canonical_equals).into())
    }

    fn return_dtype(&self, _args: &InvocationArgs) -> VortexResult<DType> {
        Ok(DType::Bool(Nullability::NonNullable))
    }

    fn return_len(&self, _args: &InvocationArgs) -> VortexResult<usize> {
        Ok(1)
    }

    fn is_elementwise(&self) -> bool {
        false
    }
}

// todo: statistics
pub trait ArrayEqualsKernel: VTable {
    fn compare_array(
        &self,
        array: &Self::Array,
        other: &dyn Array,
        ignore_nullability: bool,
    ) -> VortexResult<Option<bool>>;
}

struct ArrayEqualsArgs<'a> {
    left: &'a dyn Array,
    right: &'a dyn Array,
    ignore_nullability: bool,
}

impl<'a> TryFrom<&InvocationArgs<'a>> for ArrayEqualsArgs<'a> {
    type Error = VortexError;

    fn try_from(value: &InvocationArgs<'a>) -> Result<Self, Self::Error> {
        if value.inputs.len() != 2 {
            vortex_bail!(
                "ArrayEquals function requires two arguments, got {}",
                value.inputs.len()
            );
        }
        let left = value.inputs[0]
            .array()
            .ok_or_else(|| vortex_err!("First argument must be an array"))?;

        let right = value.inputs[1]
            .array()
            .ok_or_else(|| vortex_err!("Second argument must be an array"))?;

        let options = value
            .options
            .as_any()
            .downcast_ref::<ArrayEqualsOptions>()
            .ok_or_else(|| vortex_err!("Invalid options type for array equals function"))?;

        Ok(ArrayEqualsArgs {
            left,
            right,
            ignore_nullability: options.ignore_nullability,
        })
    }
}

#[derive(Debug)]
pub struct ArrayEqualsKernelAdapter<V: VTable>(pub V);

pub struct ArrayEqualsKernelRef(ArcRef<dyn Kernel>);
inventory::collect!(ArrayEqualsKernelRef);

impl<V: VTable + ArrayEqualsKernel> ArrayEqualsKernelAdapter<V> {
    pub const fn lift(&'static self) -> ArrayEqualsKernelRef {
        ArrayEqualsKernelRef(ArcRef::new_ref(self))
    }
}

impl<V: VTable + ArrayEqualsKernel> Kernel for ArrayEqualsKernelAdapter<V> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let ArrayEqualsArgs {
            left,
            right,
            ignore_nullability,
        } = ArrayEqualsArgs::try_from(args)?;

        let Some(left) = left.as_opt::<V>() else {
            return Ok(None);
        };

        let is_equal = V::compare_array(&self.0, left, right, ignore_nullability)?;
        Ok(is_equal.map(|b| Scalar::from(b).into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrays::{BoolArray, ConstantArray, PrimitiveArray, VarBinArray};
    use crate::validity::Validity;
    use vortex_dtype::{DType, Nullability};

    #[test]
    fn test_simple_equals() {
        let arr1 = PrimitiveArray::from_iter(vec![1i32, 2, 3, 4, 5]);
        let arr2 = PrimitiveArray::from_iter(vec![1i32, 2, 3, 4, 5]);
        let arr3 = PrimitiveArray::from_iter(vec![1i32, 2, 3, 4, 6]);

        assert!(array_equals(arr1.as_ref(), arr2.as_ref()).unwrap());
        assert!(!array_equals(arr1.as_ref(), arr3.as_ref()).unwrap());
    }

    #[test]
    fn test_stats_comparison() {
        // Arrays with different stats should be detected as different early
        let arr1 = PrimitiveArray::from_iter(vec![1i32, 2, 3, 4, 5]);
        let arr2 = PrimitiveArray::from_iter(vec![10i32, 20, 30, 40, 50]);

        assert!(!array_equals(arr1.as_ref(), arr2.as_ref()).unwrap());
    }

    #[test]
    fn test_constant_arrays() {
        let const1 = ConstantArray::new(Scalar::from(42i32), 100);
        let const2 = ConstantArray::new(Scalar::from(42i32), 100);
        let const3 = ConstantArray::new(Scalar::from(43i32), 100);

        assert!(array_equals(const1.as_ref(), const2.as_ref()).unwrap());
        assert!(!array_equals(const1.as_ref(), const3.as_ref()).unwrap());
    }

    #[test]
    fn test_different_types() {
        let int_arr = PrimitiveArray::from_iter(vec![1i32, 2, 3]);
        let float_arr = PrimitiveArray::from_iter(vec![1.0f32, 2.0, 3.0]);

        assert!(!array_equals(int_arr.as_ref(), float_arr.as_ref()).unwrap());
    }

    #[test]
    fn test_with_nulls() {
        let arr1 = PrimitiveArray::from_option_iter(vec![Some(1i32), None, Some(3), Some(4)]);
        let arr2 = PrimitiveArray::from_option_iter(vec![Some(1i32), None, Some(3), Some(4)]);
        let arr3 = PrimitiveArray::from_option_iter(vec![Some(1i32), Some(2), Some(3), Some(4)]);

        assert!(array_equals(arr1.as_ref(), arr2.as_ref()).unwrap());
        assert!(!array_equals(arr1.as_ref(), arr3.as_ref()).unwrap());
    }

    #[test]
    fn test_null_arrays() {
        let arr1 = PrimitiveArray::from_option_iter(vec![None::<i32>, None, None]);
        let arr2 = PrimitiveArray::from_option_iter(vec![None::<i32>, None, None]);

        assert!(array_equals(arr1.as_ref(), arr2.as_ref()).unwrap());
    }

    #[test]
    fn test_bool_arrays() {
        use arrow_buffer::BooleanBuffer;

        let arr1 = BoolArray::new(
            BooleanBuffer::from_iter([true, false, true, false]),
            Validity::AllValid,
        );
        let arr2 = BoolArray::new(
            BooleanBuffer::from_iter([true, false, true, false]),
            Validity::AllValid,
        );
        let arr3 = BoolArray::new(
            BooleanBuffer::from_iter([true, false, false, false]),
            Validity::AllValid,
        );

        assert!(array_equals(arr1.as_ref(), arr2.as_ref()).unwrap());
        assert!(!array_equals(arr1.as_ref(), arr3.as_ref()).unwrap());
    }

    #[test]
    fn test_empty_arrays() {
        let empty1 = PrimitiveArray::from_iter(Vec::<i32>::new());
        let empty2 = PrimitiveArray::from_iter(Vec::<i32>::new());

        assert!(array_equals(empty1.as_ref(), empty2.as_ref()).unwrap());
    }

    #[test]
    fn test_different_lengths() {
        let arr1 = PrimitiveArray::from_iter(vec![1i32, 2, 3]);
        let arr2 = PrimitiveArray::from_iter(vec![1i32, 2, 3, 4]);

        assert!(!array_equals(arr1.as_ref(), arr2.as_ref()).unwrap());
    }

    #[test]
    fn test_large_arrays() {
        // Test arrays larger than BATCH_SIZE
        let data1: Vec<i64> = (0..100_000).collect();
        let data2: Vec<i64> = (0..100_000).collect();
        let mut data3 = data1.clone();
        data3[99_999] = 999_999;

        let arr1 = PrimitiveArray::from_iter(data1);
        let arr2 = PrimitiveArray::from_iter(data2);
        let arr3 = PrimitiveArray::from_iter(data3);

        assert!(array_equals(arr1.as_ref(), arr2.as_ref()).unwrap());
        assert!(!array_equals(arr1.as_ref(), arr3.as_ref()).unwrap());
    }

    #[test]
    fn test_non_canonical_arrays() {
        let varbin1 = VarBinArray::from_vec(
            vec!["hello".as_bytes(), "world".as_bytes()],
            DType::Utf8(Nullability::NonNullable),
        );
        let varbin2 = VarBinArray::from_vec(
            vec!["hello".as_bytes(), "world".as_bytes()],
            DType::Utf8(Nullability::NonNullable),
        );
        let varbin3 = VarBinArray::from_vec(
            vec!["hello".as_bytes(), "earth".as_bytes()],
            DType::Utf8(Nullability::NonNullable),
        );

        assert!(array_equals(varbin1.as_ref(), varbin2.as_ref()).unwrap());
        assert!(!array_equals(varbin1.as_ref(), varbin3.as_ref()).unwrap());
    }
}
