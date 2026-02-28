// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;
use std::ops::BitAnd;

use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::Direction;
use crate::ArrayRef;
use crate::IntoArray;
use crate::aggregate_fn::accumulator::Accumulator;
use crate::arrays::BoolArray;
use crate::canonical::ToCanonical;
use crate::scalar::Scalar;

/// Accumulator for boolean min/max.
///
/// - Min is saturated as soon as `false` is seen (since false < true).
/// - Max is saturated as soon as `true` is seen.
pub(super) struct BoolExtremumAccumulator<D> {
    current: Option<bool>,
    results: Vec<Option<bool>>,
    _direction: PhantomData<D>,
}

impl<D: Direction> BoolExtremumAccumulator<D> {
    pub(super) fn new() -> Self {
        Self {
            current: None,
            results: Vec::new(),
            _direction: PhantomData,
        }
    }

    #[inline]
    fn consider(&mut self, candidate: bool) {
        match self.current {
            None => self.current = Some(candidate),
            Some(cur) => {
                if D::should_replace_bool(cur, candidate) {
                    self.current = Some(candidate);
                }
            }
        }
    }
}

/// Count of true and false values in a boolean array, considering validity.
struct BoolCounts {
    true_count: u64,
    false_count: u64,
}

fn bool_counts(bool_array: &BoolArray) -> VortexResult<BoolCounts> {
    let validity = bool_array.validity_mask()?;
    let bits = bool_array.to_bit_buffer();

    match &validity {
        Mask::AllTrue(_) => {
            let true_count = bits.true_count() as u64;
            let false_count = bool_array.len() as u64 - true_count;
            Ok(BoolCounts {
                true_count,
                false_count,
            })
        }
        Mask::AllFalse(_) => Ok(BoolCounts {
            true_count: 0,
            false_count: 0,
        }),
        Mask::Values(v) => {
            let valid_bits = bits.bitand(v.bit_buffer());
            let true_count = valid_bits.true_count() as u64;
            let valid_count = v.bit_buffer().true_count() as u64;
            let false_count = valid_count - true_count;
            Ok(BoolCounts {
                true_count,
                false_count,
            })
        }
    }
}

impl<D: Direction> Accumulator for BoolExtremumAccumulator<D> {
    fn accumulate(&mut self, batch: &ArrayRef) -> VortexResult<()> {
        let bool_array = batch.to_bool();
        let counts = bool_counts(&bool_array)?;

        if counts.true_count > 0 {
            self.consider(true);
        }
        if counts.false_count > 0 {
            self.consider(false);
        }

        Ok(())
    }

    fn merge(&mut self, state: &Scalar) -> VortexResult<()> {
        if state.is_null() {
            return Ok(());
        }
        if let Some(v) = state.as_bool().value() {
            self.consider(v);
        }
        Ok(())
    }

    fn is_saturated(&self) -> bool {
        self.current.is_some_and(D::is_saturated_bool)
    }

    fn flush(&mut self) -> VortexResult<()> {
        self.results.push(self.current);
        self.current = None;
        Ok(())
    }

    fn finish(self: Box<Self>) -> VortexResult<ArrayRef> {
        Ok(BoolArray::from_iter(self.results).into_array())
    }
}
