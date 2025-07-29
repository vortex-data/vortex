// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::LazyLock;

use arcref::ArcRef;
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_scalar::Scalar;

use crate::Array;
use crate::arrays::ConstantArray;
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
            options: &ArrayEqualsOptions {
                ignore_nullability,
                batch_size: None,
            },
        })?
        .unwrap_scalar()?
        .as_bool()
        .value()
        .vortex_expect("non-nullable"))
}

#[derive(Clone, Copy)]
struct ArrayEqualsOptions {
    ignore_nullability: bool,
    batch_size: Option<usize>,
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
            batch_size,
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

        // Early return for empty arrays - they're equal regardless of type
        if left.is_empty() {
            return Ok(Scalar::from(true).into());
        }

        // Handle constant array comparisons
        match (left.as_constant(), right.as_constant()) {
            (Some(l_scalar), Some(r_scalar)) => {
                // Both are constants - compare scalars directly
                return Ok(Scalar::from(l_scalar.eq(&r_scalar)).into());
            }
            (Some(constant), None) | (None, Some(constant)) => {
                // One is constant, one is not - they can only be equal if all elements
                // of the non-constant array equal the constant
                let non_constant_array = if left.as_constant().is_some() {
                    right
                } else {
                    left
                };

                // Quick check using statistics
                if constant.is_null() {
                    // All elements must be null for equality
                    if let Some(Precision::Exact(null_count_value)) =
                        non_constant_array.statistics().get(Stat::NullCount)
                    {
                        let null_count_scalar = Scalar::new(
                            DType::Primitive(vortex_dtype::PType::U64, Nullability::NonNullable),
                            null_count_value,
                        );
                        if let Ok(Some(count)) = null_count_scalar.as_primitive().as_::<usize>() {
                            return Ok(Scalar::from(count == non_constant_array.len()).into());
                        }
                    }
                } else {
                    // Non-null constant - check if min/max statistics can rule out equality
                    let stats = non_constant_array.statistics();
                    if let (Some(Precision::Exact(min)), Some(Precision::Exact(max))) =
                        (stats.get(Stat::Min), stats.get(Stat::Max))
                    {
                        let min_scalar = Scalar::new(non_constant_array.dtype().clone(), min);
                        let max_scalar = Scalar::new(non_constant_array.dtype().clone(), max);
                        if !constant.eq(&min_scalar) || !constant.eq(&max_scalar) {
                            return Ok(Scalar::from(false).into());
                        }
                    }
                }

                // Use compare function to check if all elements equal the constant
                // Create a constant array of the same length for comparison
                let constant_array = ConstantArray::new(constant, non_constant_array.len());
                let compare_result =
                    compare(non_constant_array, constant_array.as_ref(), Operator::Eq)?;

                // Check if all comparison results are true (all elements equal the constant)
                if let Some(all_equal) = check_constant_result(&compare_result)? {
                    return Ok(Scalar::from(all_equal).into());
                }

                // Check via statistics if possible
                if let Some(all_true) = check_comparison_stats(&compare_result) {
                    return Ok(Scalar::from(all_true).into());
                }

                // Fall through to general case handling below
            }
            (None, None) => {
                // Neither is constant - continue with general algorithm
            }
        }

        // Check statistics for early exit
        // TODO(optimization): Add more sophisticated statistical comparisons for floating point arrays
        if !check_stats_equality(left, right) {
            return Ok(Scalar::from(false).into());
        }

        let args = InvocationArgs {
            inputs: &[left.into(), right.into()],
            options: &ArrayEqualsOptions {
                ignore_nullability,
                batch_size,
            },
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
            options: &ArrayEqualsOptions {
                ignore_nullability,
                batch_size,
            },
        };
        if let Some(output) = right.invoke(&ARRAY_EQUALS_FN, &swapped_args)? {
            return Ok(output);
        }

        // Try canonical arrays if not already canonical
        if !left.is_canonical() || !right.is_canonical() {
            log::debug!(
                "Falling back to canonical array_equals for encodings {} and {}",
                left.encoding_id(),
                right.encoding_id()
            );

            let left_canonical = left.to_canonical()?;
            let right_canonical = right.to_canonical()?;

            return Ok(Scalar::from(array_equals_opts(
                left_canonical.as_ref(),
                right_canonical.as_ref(),
                ignore_nullability,
            )?)
            .into());
        }

        // Final fallback to chunked comparison for canonical arrays
        log::debug!(
            "Using chunked comparison fallback for canonical arrays {} and {}",
            left.encoding_id(),
            right.encoding_id()
        );

        let all_equal = compare_chunked(left, right, batch_size)?;
        Ok(Scalar::from(all_equal).into())
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
    batch_size: Option<usize>,
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
            batch_size: options.batch_size,
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
            batch_size: _, // Not used in kernel adapters
        } = ArrayEqualsArgs::try_from(args)?;

        let Some(left) = left.as_opt::<V>() else {
            return Ok(None);
        };

        let is_equal = V::compare_array(&self.0, left, right, ignore_nullability)?;
        Ok(is_equal.map(|b| Scalar::from(b).into()))
    }
}

/// Compare arrays in chunks to avoid loading entire arrays into memory
fn compare_chunked(
    left: &dyn Array,
    right: &dyn Array,
    batch_size: Option<usize>,
) -> VortexResult<bool> {
    const DEFAULT_BATCH_SIZE: usize = 65536; // 64K elements per batch
    let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE);

    let mut offset = 0;
    while offset < left.len() {
        let end = (offset + batch_size).min(left.len());

        let left_slice = left.slice(offset, end)?;
        let right_slice = right.slice(offset, end)?;

        if !compare_batch(&left_slice, &right_slice)? {
            return Ok(false);
        }

        offset = end;
    }

    Ok(true)
}

/// Compare a single batch of arrays
fn compare_batch(left: &dyn Array, right: &dyn Array) -> VortexResult<bool> {
    let compare_result = compare(left, right, Operator::Eq)?;

    // Check if the comparison result indicates all equal
    if let Some(all_equal) = check_constant_result(&compare_result)? {
        return Ok(all_equal);
    }

    // Not constant - need to check each value
    check_non_constant_result(&compare_result, left, right)
}

/// Check if a constant comparison result indicates equality
fn check_constant_result(compare_result: &dyn Array) -> VortexResult<Option<bool>> {
    if let Some(constant_scalar) = compare_result.as_constant() {
        // If constant is true, all are equal
        Ok(Some(
            constant_scalar.is_valid() && constant_scalar.as_bool().value() == Some(true),
        ))
    } else {
        Ok(None)
    }
}

/// Check non-constant comparison results, handling null comparisons
fn check_non_constant_result(
    compare_result: &dyn Array,
    left: &dyn Array,
    right: &dyn Array,
) -> VortexResult<bool> {
    // First, check statistics for quick rejection
    if let Some(all_true) = check_comparison_stats(compare_result) {
        return Ok(all_true);
    }

    // Fallback to element-wise check
    for i in 0..compare_result.len() {
        let cmp_scalar = compare_result.scalar_at(i)?;

        // Check for definite inequality
        if cmp_scalar.is_valid() && cmp_scalar.as_bool().value() == Some(false) {
            return Ok(false);
        }

        // Handle null comparison results
        if cmp_scalar.is_null() && !check_null_equality(left, right, i)? {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Check comparison statistics for quick determination
fn check_comparison_stats(compare_result: &dyn Array) -> Option<bool> {
    // If min is false, we have at least one false
    if let Some(Precision::Exact(min)) = compare_result.statistics().get(Stat::Min) {
        if min.as_bool().ok()? == Some(false) {
            return Some(false);
        }
    }

    // If both min and max are true, all are true
    if let Some(Precision::Exact(min)) = compare_result.statistics().get(Stat::Min) {
        if let Some(Precision::Exact(max)) = compare_result.statistics().get(Stat::Max) {
            if min.as_bool().ok()? == Some(true) && max.as_bool().ok()? == Some(true) {
                return Some(true);
            }
        }
    }

    None
}

/// Check if two potentially null values at a given index are equal
fn check_null_equality(left: &dyn Array, right: &dyn Array, index: usize) -> VortexResult<bool> {
    let left_val = left.scalar_at(index)?;
    let right_val = right.scalar_at(index)?;

    // Both null or both non-null means they could be equal
    // (if both non-null, the comparison would have returned true/false, not null)
    Ok(left_val.is_null() == right_val.is_null())
}

/// Check statistics equality for early exit
fn check_stats_equality(left: &dyn Array, right: &dyn Array) -> bool {
    let stats_to_check = [
        Stat::IsConstant,
        Stat::IsSorted,
        Stat::IsStrictSorted,
        Stat::Max,
        Stat::Min,
        Stat::Sum,
        Stat::NullCount,
        Stat::NaNCount,
    ];

    for stat in stats_to_check {
        match (left.statistics().get(stat), right.statistics().get(stat)) {
            (Some(Precision::Exact(left_v)), Some(Precision::Exact(right_v))) => {
                if !left_v.eq(&right_v) {
                    return false;
                }
            }
            _ => continue,
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IntoArray;
    use crate::arrays::{BoolArray, ChunkedArray, ConstantArray, PrimitiveArray, VarBinArray};
    use crate::validity::Validity;
    use vortex_dtype::{DType, Nullability, PType};

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

    #[test]
    fn test_float_precision() {
        // Test if statistics-based comparison can handle float precision issues
        let arr1 = PrimitiveArray::from_iter(vec![1.0f64, 2.0, 3.0, 4.0, 5.0]);
        let arr2 = PrimitiveArray::from_iter(vec![1.0f64, 2.0, 3.0, 4.0, 5.0]);

        // Arrays with exact same values should be equal
        assert!(array_equals(arr1.as_ref(), arr2.as_ref()).unwrap());

        // Arrays with slightly different values should not be equal
        let arr3 = PrimitiveArray::from_iter(vec![1.0f64, 2.0, 3.0, 4.0, 5.0000000001]);
        assert!(!array_equals(arr1.as_ref(), arr3.as_ref()).unwrap());
    }

    #[test]
    fn test_batch_size_functionality() {
        // Test arrays larger than default batch size with different batch sizes
        let data1: Vec<i32> = (0..150_000).collect();
        let data2: Vec<i32> = (0..150_000).collect();

        let arr1 = PrimitiveArray::from_iter(data1);
        let arr2 = PrimitiveArray::from_iter(data2);

        // Test with different batch sizes (though we can't pass batch_size directly in public API)
        assert!(array_equals(arr1.as_ref(), arr2.as_ref()).unwrap());
    }

    #[test]
    fn test_primitive_vs_dict_array() {
        // Test comparing primitive array with dictionary-encoded array containing same values

        let primitive_arr = PrimitiveArray::from_iter(vec![1i32, 2, 1, 3, 2, 1]);

        // Create a chunked array as a proxy for non-canonical encoding
        let chunk1 = PrimitiveArray::from_iter(vec![1i32, 2, 1]);
        let chunk2 = PrimitiveArray::from_iter(vec![3i32, 2, 1]);
        let chunked_arr = ChunkedArray::try_new(
            vec![chunk1.into_array(), chunk2.into_array()],
            primitive_arr.dtype().clone(),
        )
        .unwrap();

        // Should be equal as they contain the same logical values
        assert!(array_equals(primitive_arr.as_ref(), chunked_arr.as_ref()).unwrap());

        // Test with different values
        let chunk1_copy = PrimitiveArray::from_iter(vec![1i32, 2, 1]);
        let different_chunk2 = PrimitiveArray::from_iter(vec![3i32, 2, 4]);
        let different_chunked = ChunkedArray::try_new(
            vec![chunk1_copy.into_array(), different_chunk2.into_array()],
            primitive_arr.dtype().clone(),
        )
        .unwrap();

        assert!(!array_equals(primitive_arr.as_ref(), different_chunked.as_ref()).unwrap());
    }

    #[test]
    fn test_constant_null_arrays() {
        // Test constant null arrays - should be equal to each other but not to non-null constants
        let null_const1 = ConstantArray::new(
            Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
            5,
        );
        let null_const2 = ConstantArray::new(
            Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
            5,
        );
        let non_null_const = ConstantArray::new(Scalar::from(42i32), 5);

        // Both null constants should be equal
        assert!(array_equals(null_const1.as_ref(), null_const2.as_ref()).unwrap());

        // Null constant should not equal non-null constant
        assert!(!array_equals(null_const1.as_ref(), non_null_const.as_ref()).unwrap());
        assert!(!array_equals(non_null_const.as_ref(), null_const1.as_ref()).unwrap());
    }

    #[test]
    fn test_mixed_constant_non_constant() {
        // Test comparing constant arrays with non-constant arrays
        let constant_42 = ConstantArray::new(Scalar::from(42i32), 4);
        let all_42s = PrimitiveArray::from_iter(vec![42i32, 42, 42, 42]);
        let mixed_values = PrimitiveArray::from_iter(vec![42i32, 42, 43, 42]);

        // Constant should equal array with all same values
        assert!(array_equals(constant_42.as_ref(), all_42s.as_ref()).unwrap());
        assert!(array_equals(all_42s.as_ref(), constant_42.as_ref()).unwrap());

        // Constant should not equal array with different values
        assert!(!array_equals(constant_42.as_ref(), mixed_values.as_ref()).unwrap());
        assert!(!array_equals(mixed_values.as_ref(), constant_42.as_ref()).unwrap());

        // Test with null constant
        let null_constant = ConstantArray::new(
            Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
            3,
        );
        let all_nulls = PrimitiveArray::from_option_iter(vec![None::<i32>, None, None]);
        let mixed_nulls = PrimitiveArray::from_option_iter(vec![None::<i32>, Some(42), None]);

        // Null constant should equal array with all nulls
        assert!(array_equals(null_constant.as_ref(), all_nulls.as_ref()).unwrap());
        assert!(array_equals(all_nulls.as_ref(), null_constant.as_ref()).unwrap());

        // Null constant should not equal array with mixed nulls and values
        assert!(!array_equals(null_constant.as_ref(), mixed_nulls.as_ref()).unwrap());
        assert!(!array_equals(mixed_nulls.as_ref(), null_constant.as_ref()).unwrap());
    }
}
