// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::CheckedAdd;
use num_traits::ToPrimitive;
use num_traits::WrappingAdd;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::aggregate_fn::accumulator::Accumulator;
use crate::arrays::PrimitiveArray;
use crate::canonical::ToCanonical;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;
use crate::scalar::Scalar;

pub(super) struct IntSumAccumulator<R> {
    sum: R,
    overflowed: bool,
    /// Whether at least one non-null value has been accumulated.
    has_values: bool,
    /// Whether accumulate() or merge() has been called at all (even with all-null data).
    has_input: bool,
    checked: bool,
    results: Vec<Option<R>>,
}

impl<R: NativePType + CheckedAdd + WrappingAdd> IntSumAccumulator<R> {
    pub(super) fn new(checked: bool) -> Self {
        Self {
            sum: R::default(),
            overflowed: false,
            has_values: false,
            has_input: false,
            checked,
            results: Vec::new(),
        }
    }
}

fn accumulate_all_valid<T: NativePType + ToPrimitive, R: NativePType + CheckedAdd + WrappingAdd>(
    values: &[T],
    sum: &mut R,
    overflowed: &mut bool,
    has_values: &mut bool,
    checked: bool,
) {
    for &v in values {
        if *overflowed {
            return;
        }
        *has_values = true;
        if checked {
            let Some(widened) = R::from(v) else {
                *overflowed = true;
                return;
            };
            let Some(new_sum) = sum.checked_add(&widened) else {
                *overflowed = true;
                return;
            };
            *sum = new_sum;
        } else {
            let widened = R::from(v).unwrap_or_default();
            *sum = sum.wrapping_add(&widened);
        }
    }
}

fn accumulate_with_mask<T: NativePType + ToPrimitive, R: NativePType + CheckedAdd + WrappingAdd>(
    values: &[T],
    mask: &vortex_mask::MaskValues,
    sum: &mut R,
    overflowed: &mut bool,
    has_values: &mut bool,
    checked: bool,
) {
    for (&v, valid) in values.iter().zip(mask.bit_buffer().iter()) {
        if *overflowed {
            return;
        }
        if valid {
            *has_values = true;
            if checked {
                let Some(widened) = R::from(v) else {
                    *overflowed = true;
                    return;
                };
                let Some(new_sum) = sum.checked_add(&widened) else {
                    *overflowed = true;
                    return;
                };
                *sum = new_sum;
            } else {
                let widened = R::from(v).unwrap_or_default();
                *sum = sum.wrapping_add(&widened);
            }
        }
    }
}

impl<R: NativePType + CheckedAdd + WrappingAdd> Accumulator for IntSumAccumulator<R> {
    fn accumulate(&mut self, batch: &ArrayRef) -> VortexResult<()> {
        self.has_input = true;
        let primitive = batch.to_primitive();
        let validity = primitive.validity_mask()?;

        match_each_native_ptype!(primitive.ptype(), integral: |T| {
            let values = primitive.as_slice::<T>();
            match &validity {
                Mask::AllTrue(_) => accumulate_all_valid(
                    values,
                    &mut self.sum,
                    &mut self.overflowed,
                    &mut self.has_values,
                    self.checked,
                ),
                Mask::AllFalse(_) => {}
                Mask::Values(v) => accumulate_with_mask(
                    values,
                    v,
                    &mut self.sum,
                    &mut self.overflowed,
                    &mut self.has_values,
                    self.checked,
                ),
            }
        }, floating: |_T| {
            unreachable!("IntSumAccumulator should not be used with floating-point types");
        });

        Ok(())
    }

    fn merge(&mut self, state: &Scalar) -> VortexResult<()> {
        if state.is_null() {
            return Ok(());
        }
        self.has_input = true;
        let val = state.as_primitive().typed_value::<R>();
        if let Some(v) = val {
            self.has_values = true;
            if self.checked {
                if let Some(new_sum) = self.sum.checked_add(&v) {
                    self.sum = new_sum;
                } else {
                    self.overflowed = true;
                }
            } else {
                self.sum = self.sum.wrapping_add(&v);
            }
        }
        Ok(())
    }

    fn is_saturated(&self) -> bool {
        self.checked && self.overflowed
    }

    fn flush(&mut self) -> VortexResult<()> {
        let result = if self.overflowed {
            None
        } else if self.has_values {
            Some(self.sum)
        } else if self.has_input {
            // All-null group: no non-null values were seen.
            None
        } else {
            // Empty group: no accumulate/merge calls at all. Identity is zero.
            Some(R::default())
        };
        self.results.push(result);
        self.sum = R::default();
        self.overflowed = false;
        self.has_values = false;
        self.has_input = false;
        Ok(())
    }

    fn finish(self: Box<Self>) -> VortexResult<ArrayRef> {
        Ok(PrimitiveArray::from_option_iter(self.results).into_array())
    }
}
