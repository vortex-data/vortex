// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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

pub(super) struct FloatSumAccumulator {
    sum: f64,
    /// Whether at least one non-null value has been accumulated.
    has_values: bool,
    /// Whether accumulate() or merge() has been called at all (even with all-null data).
    has_input: bool,
    results: Vec<Option<f64>>,
}

impl FloatSumAccumulator {
    pub(super) fn new() -> Self {
        Self {
            sum: 0.0,
            has_values: false,
            has_input: false,
            results: Vec::new(),
        }
    }
}

fn accumulate_all_valid<T: NativePType>(values: &[T], sum: &mut f64, has_values: &mut bool) {
    for v in values {
        *has_values = true;
        *sum += v.to_f64().unwrap_or(0.0);
    }
}

fn accumulate_with_mask<T: NativePType>(
    values: &[T],
    mask: &vortex_mask::MaskValues,
    sum: &mut f64,
    has_values: &mut bool,
) {
    for (v, valid) in values.iter().zip(mask.bit_buffer().iter()) {
        if valid {
            *has_values = true;
            *sum += v.to_f64().unwrap_or(0.0);
        }
    }
}

impl Accumulator for FloatSumAccumulator {
    fn accumulate(&mut self, batch: &ArrayRef) -> VortexResult<()> {
        self.has_input = true;
        let primitive = batch.to_primitive();
        let validity = primitive.validity_mask()?;

        match_each_native_ptype!(primitive.ptype(), integral: |_T| {
            unreachable!("FloatSumAccumulator should not be used with integer types");
        }, floating: |T| {
            let values = primitive.as_slice::<T>();
            match &validity {
                Mask::AllTrue(_) => accumulate_all_valid(
                    values,
                    &mut self.sum,
                    &mut self.has_values,
                ),
                Mask::AllFalse(_) => {}
                Mask::Values(v) => accumulate_with_mask(
                    values,
                    v,
                    &mut self.sum,
                    &mut self.has_values,
                ),
            }
        });

        Ok(())
    }

    fn merge(&mut self, state: &Scalar) -> VortexResult<()> {
        if state.is_null() {
            return Ok(());
        }
        self.has_input = true;
        if let Some(v) = state.as_primitive().typed_value::<f64>() {
            self.has_values = true;
            self.sum += v;
        }
        Ok(())
    }

    fn flush(&mut self) -> VortexResult<()> {
        let result = if self.has_values {
            Some(self.sum)
        } else if self.has_input {
            // All-null group.
            None
        } else {
            // Empty group: identity is zero.
            Some(0.0)
        };
        self.results.push(result);
        self.sum = 0.0;
        self.has_values = false;
        self.has_input = false;
        Ok(())
    }

    fn finish(self: Box<Self>) -> VortexResult<ArrayRef> {
        Ok(PrimitiveArray::from_option_iter(self.results).into_array())
    }
}
