// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::aggregate_fn::accumulator::Accumulator;
use crate::arrays::PrimitiveArray;
use crate::canonical::ToCanonical;
use crate::scalar::Scalar;

/// Accumulator that sums boolean values by counting `true` as 1 and `false` as 0.
///
/// Output type is `u64` (nullable). Overflow is theoretically possible but extremely
/// unlikely since it would require more than `u64::MAX` true values.
pub(super) struct BoolSumAccumulator {
    count: u64,
    /// Whether at least one non-null value has been accumulated.
    has_values: bool,
    /// Whether accumulate() or merge() has been called at all (even with all-null data).
    has_input: bool,
    checked: bool,
    overflowed: bool,
    results: Vec<Option<u64>>,
}

impl BoolSumAccumulator {
    pub(super) fn new(checked: bool) -> Self {
        Self {
            count: 0,
            has_values: false,
            has_input: false,
            checked,
            overflowed: false,
            results: Vec::new(),
        }
    }
}

impl Accumulator for BoolSumAccumulator {
    fn accumulate(&mut self, batch: &ArrayRef) -> VortexResult<()> {
        self.has_input = true;
        if self.overflowed {
            return Ok(());
        }

        let bool_array = batch.to_bool();
        let validity = bool_array.validity_mask()?;

        let true_count = match &validity {
            Mask::AllTrue(_) => bool_array.to_bit_buffer().true_count() as u64,
            Mask::AllFalse(_) => return Ok(()),
            Mask::Values(v) => bool_array
                .to_bit_buffer()
                .bitand(v.bit_buffer())
                .true_count() as u64,
        };

        self.has_values = true;
        if self.checked {
            if let Some(new_count) = self.count.checked_add(true_count) {
                self.count = new_count;
            } else {
                self.overflowed = true;
            }
        } else {
            self.count = self.count.wrapping_add(true_count);
        }

        Ok(())
    }

    fn merge(&mut self, state: &Scalar) -> VortexResult<()> {
        if state.is_null() {
            return Ok(());
        }
        self.has_input = true;
        if let Some(v) = state.as_primitive().typed_value::<u64>() {
            self.has_values = true;
            if self.checked {
                if let Some(new_count) = self.count.checked_add(v) {
                    self.count = new_count;
                } else {
                    self.overflowed = true;
                }
            } else {
                self.count = self.count.wrapping_add(v);
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
            Some(self.count)
        } else if self.has_input {
            // All-null group.
            None
        } else {
            // Empty group: identity is zero.
            Some(0)
        };
        self.results.push(result);
        self.count = 0;
        self.has_values = false;
        self.has_input = false;
        self.overflowed = false;
        Ok(())
    }

    fn finish(self: Box<Self>) -> VortexResult<ArrayRef> {
        Ok(PrimitiveArray::from_option_iter(self.results).into_array())
    }
}
