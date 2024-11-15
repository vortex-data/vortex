use num_traits::AsPrimitive;
use vortex_array::compute::unary::{FillForwardFn, ScalarAtFn};
use vortex_array::compute::{ArrayCompute, SliceFn, TakeFn};
use vortex_array::validity::{ArrayValidity, Validity};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayDType, ArrayData, IntoArrayData};
use vortex_dtype::{match_each_integer_ptype, Nullability};
use vortex_error::{vortex_err, VortexResult};
use vortex_scalar::Scalar;

use super::ByteBoolArray;

impl ArrayCompute for ByteBoolArray {
    fn fill_forward(&self) -> Option<&dyn FillForwardFn> {
        None
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

impl ScalarAtFn for ByteBoolArray {
    fn scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        Ok(self.scalar_at_unchecked(index))
    }

    fn scalar_at_unchecked(&self, index: usize) -> Scalar {
        Scalar::bool(self.buffer()[index] == 1, self.dtype().nullability())
    }
}

impl SliceFn for ByteBoolArray {
    fn slice(&self, start: usize, stop: usize) -> VortexResult<ArrayData> {
        Ok(ByteBoolArray::try_new(
            self.buffer().slice(start..stop),
            self.validity().slice(start, stop)?,
        )?
        .into_array())
    }
}

impl TakeFn for ByteBoolArray {
    fn take(&self, indices: &ArrayData) -> VortexResult<ArrayData> {
        let validity = self.validity();
        let indices = indices.clone().as_primitive();
        let bools = self.maybe_null_slice();

        let arr = match validity {
            Validity::AllValid | Validity::NonNullable => {
                let bools = match_each_integer_ptype!(indices.ptype(), |$I| {
                    indices.maybe_null_slice::<$I>()
                    .iter()
                    .map(|&idx| {
                        let idx: usize = idx.as_();
                        bools[idx]
                    })
                    .collect::<Vec<_>>()
                });

                Self::from(bools).into_array()
            }
            Validity::AllInvalid => Self::from(vec![None; indices.len()]).into_array(),

            Validity::Array(_) => {
                let bools = match_each_integer_ptype!(indices.ptype(), |$I| {
                    indices.maybe_null_slice::<$I>()
                    .iter()
                    .map(|&idx| {
                        let idx = idx.as_();
                        if validity.is_valid(idx) {
                            Some(bools[idx])
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<Option<_>>>()
                });

                Self::from(bools).into_array()
            }
        };

        Ok(arr)
    }
}

impl FillForwardFn for ByteBoolArray {
    fn fill_forward(&self) -> VortexResult<ArrayData> {
        let validity = self.logical_validity();
        if self.dtype().nullability() == Nullability::NonNullable {
            return Ok(self.clone().into());
        }
        // all valid, but we need to convert to non-nullable
        if validity.all_valid() {
            return Ok(Self::try_new(self.buffer().clone(), Validity::AllValid)?.into_array());
        }
        // all invalid => fill with default value (false)
        if validity.all_invalid() {
            return Ok(
                Self::try_from_vec(vec![false; self.len()], Validity::AllValid)?.into_array(),
            );
        }

        let validity = validity
            .to_null_buffer()?
            .ok_or_else(|| vortex_err!("Failed to convert array validity to null buffer"))?;

        let bools = self.maybe_null_slice();
        let mut last_value = bool::default();

        let filled = bools
            .iter()
            .zip(validity.inner().iter())
            .map(|(&v, is_valid)| {
                if is_valid {
                    last_value = v
                }

                last_value
            })
            .collect::<Vec<_>>();

        Ok(Self::try_from_vec(filled, Validity::AllValid)?.into_array())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::compute::unary::{scalar_at, scalar_at_unchecked};
    use vortex_array::compute::{compare, slice, Operator};
    use vortex_scalar::ScalarValue;

    use super::*;

    #[test]
    fn test_slice() {
        let original = vec![Some(true), Some(true), None, Some(false), None];
        let vortex_arr = ByteBoolArray::from(original.clone());

        let sliced_arr = slice(vortex_arr.as_ref(), 1, 4).unwrap();
        let sliced_arr = ByteBoolArray::try_from(sliced_arr).unwrap();

        let s = scalar_at_unchecked(sliced_arr.as_ref(), 0);
        assert_eq!(s.into_value().as_bool().unwrap(), Some(true));

        let s = scalar_at(sliced_arr.as_ref(), 1).unwrap();
        assert!(!sliced_arr.is_valid(1));
        assert!(s.is_null());
        assert_eq!(s.into_value().as_bool().unwrap(), None);

        let s = scalar_at_unchecked(sliced_arr.as_ref(), 2);
        assert_eq!(s.into_value().as_bool().unwrap(), Some(false));
    }

    #[test]
    fn test_compare_all_equal() {
        let lhs = ByteBoolArray::from(vec![true; 5]);
        let rhs = ByteBoolArray::from(vec![true; 5]);

        let arr = compare(lhs.as_ref(), rhs.as_ref(), Operator::Eq).unwrap();

        for i in 0..arr.len() {
            let s = scalar_at_unchecked(arr.as_ref(), i);
            assert!(s.is_valid());
            assert_eq!(s.value(), &ScalarValue::Bool(true));
        }
    }

    #[test]
    fn test_compare_all_different() {
        let lhs = ByteBoolArray::from(vec![false; 5]);
        let rhs = ByteBoolArray::from(vec![true; 5]);

        let arr = compare(lhs.as_ref(), rhs.as_ref(), Operator::Eq).unwrap();

        for i in 0..arr.len() {
            let s = scalar_at(&arr, i).unwrap();
            assert!(s.is_valid());
            assert_eq!(s.value(), &ScalarValue::Bool(false));
        }
    }

    #[test]
    fn test_compare_with_nulls() {
        let lhs = ByteBoolArray::from(vec![true; 5]);
        let rhs = ByteBoolArray::from(vec![Some(true), Some(true), Some(true), Some(false), None]);

        let arr = compare(lhs.as_ref(), rhs.as_ref(), Operator::Eq).unwrap();

        for i in 0..3 {
            let s = scalar_at(&arr, i).unwrap();
            assert!(s.is_valid());
            assert_eq!(s.value(), &ScalarValue::Bool(true));
        }

        let s = scalar_at(&arr, 3).unwrap();
        assert!(s.is_valid());
        assert_eq!(s.value(), &ScalarValue::Bool(false));

        let s = scalar_at(&arr, 4).unwrap();
        assert!(s.is_null());
    }
}
