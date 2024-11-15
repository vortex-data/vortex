use arrow_buffer::bit_util::ceil;
use arrow_buffer::{BooleanBuffer, MutableBuffer};
use vortex_dtype::{match_each_native_ptype, NativePType};
use vortex_error::{vortex_err, VortexExpect, VortexResult};
use vortex_scalar::PrimitiveScalar;

use crate::array::primitive::PrimitiveArray;
use crate::array::{BoolArray, ConstantArray};
use crate::compute::{MaybeCompareFn, Operator};
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayDType, ArrayData, IntoArrayData};

impl MaybeCompareFn for PrimitiveArray {
    fn maybe_compare(
        &self,
        other: &ArrayData,
        operator: Operator,
    ) -> Option<VortexResult<ArrayData>> {
        if let Ok(const_array) = ConstantArray::try_from(other) {
            return Some(primitive_const_compare(self, const_array, operator));
        }

        if let Ok(primitive) = PrimitiveArray::try_from(other) {
            let match_mask = match_each_native_ptype!(self.ptype(), |$T| {
                apply_predicate(self.maybe_null_slice::<$T>(), primitive.maybe_null_slice::<$T>(), operator.to_fn::<$T>())
            });

            let validity = self
                .validity()
                .and(primitive.validity())
                .map(|v| v.into_nullable());

            return Some(
                validity
                    .and_then(|v| BoolArray::try_new(match_mask, v))
                    .map(|a| a.into_array()),
            );
        }

        None
    }
}

fn primitive_const_compare(
    this: &PrimitiveArray,
    other: ConstantArray,
    operator: Operator,
) -> VortexResult<ArrayData> {
    let primitive_scalar = PrimitiveScalar::try_new(other.dtype(), other.scalar_value())
        .vortex_expect("Expected a primitive scalar");

    let buffer = match_each_native_ptype!(this.ptype(), |$T| {
        let typed_value = primitive_scalar.typed_value::<$T>()
            .ok_or_else(|| vortex_err!("Type mismatch between array and constant"))?;
        primitive_value_compare::<$T>(this, typed_value, operator)
    });

    Ok(BoolArray::try_new(buffer, this.validity().into_nullable())?.into_array())
}

fn primitive_value_compare<T: NativePType>(
    this: &PrimitiveArray,
    value: T,
    op: Operator,
) -> BooleanBuffer {
    let op_fn = op.to_fn::<T>();
    let slice = this.maybe_null_slice::<T>();

    BooleanBuffer::collect_bool(this.len(), |idx| {
        op_fn(unsafe { *slice.get_unchecked(idx) }, value)
    })
}

fn apply_predicate<T: NativePType, F: Fn(T, T) -> bool>(
    lhs: &[T],
    rhs: &[T],
    f: F,
) -> BooleanBuffer {
    const BLOCK_SIZE: usize = u64::BITS as usize;

    let len = lhs.len();
    let reminder = len % BLOCK_SIZE;
    let block_count = len / BLOCK_SIZE;

    let mut buffer = MutableBuffer::new(ceil(len, BLOCK_SIZE) * 8);

    for block in 0..block_count {
        let mut packed_block = 0_u64;
        for bit_idx in 0..BLOCK_SIZE {
            let idx = bit_idx + block * BLOCK_SIZE;
            let r = f(unsafe { *lhs.get_unchecked(idx) }, unsafe {
                *rhs.get_unchecked(idx)
            });
            packed_block |= (r as u64) << bit_idx;
        }

        unsafe {
            buffer.push_unchecked(packed_block);
        }
    }

    if reminder != 0 {
        let mut packed_block = 0_u64;
        for bit_idx in 0..reminder {
            let idx = bit_idx + block_count * BLOCK_SIZE;
            let r = f(lhs[idx], rhs[idx]);
            packed_block |= (r as u64) << bit_idx;
        }

        unsafe {
            buffer.push_unchecked(packed_block);
        }
    }

    BooleanBuffer::new(buffer.into(), 0, len)
}

#[cfg(test)]
#[allow(clippy::panic_in_result_fn)]
mod test {
    use itertools::Itertools;

    use super::*;
    use crate::compute::compare;
    use crate::IntoArrayVariant;

    fn to_int_indices(indices_bits: BoolArray) -> Vec<u64> {
        let filtered = indices_bits
            .boolean_buffer()
            .iter()
            .enumerate()
            .filter_map(|(idx, v)| {
                let valid_and_true = indices_bits.validity().is_valid(idx) & v;
                valid_and_true.then_some(idx as u64)
            })
            .collect_vec();
        filtered
    }

    #[test]
    fn test_basic_comparisons() -> VortexResult<()> {
        let arr = PrimitiveArray::from_nullable_vec(vec![
            Some(1i32),
            Some(2),
            Some(3),
            Some(4),
            None,
            Some(5),
            Some(6),
            Some(7),
            Some(8),
            None,
            Some(9),
            None,
        ])
        .into_array();

        let matches = compare(&arr, &arr, Operator::Eq)?.into_bool()?;
        assert_eq!(to_int_indices(matches), [0u64, 1, 2, 3, 5, 6, 7, 8, 10]);

        let matches = compare(&arr, &arr, Operator::NotEq)?.into_bool()?;
        let empty: [u64; 0] = [];
        assert_eq!(to_int_indices(matches), empty);

        let other = PrimitiveArray::from_nullable_vec(vec![
            Some(1i32),
            Some(2),
            Some(3),
            Some(4),
            None,
            Some(6),
            Some(7),
            Some(8),
            Some(9),
            None,
            Some(10),
            None,
        ])
        .into_array();

        let matches = compare(&arr, &other, Operator::Lte)?.into_bool()?;
        assert_eq!(to_int_indices(matches), [0u64, 1, 2, 3, 5, 6, 7, 8, 10]);

        let matches = compare(&arr, &other, Operator::Lt)?.into_bool()?;
        assert_eq!(to_int_indices(matches), [5u64, 6, 7, 8, 10]);

        let matches = compare(&other, &arr, Operator::Gte)?.into_bool()?;
        assert_eq!(to_int_indices(matches), [0u64, 1, 2, 3, 5, 6, 7, 8, 10]);

        let matches = compare(&other, &arr, Operator::Gt)?.into_bool()?;
        assert_eq!(to_int_indices(matches), [5u64, 6, 7, 8, 10]);
        Ok(())
    }
}
