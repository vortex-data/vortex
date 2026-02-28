// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use num_traits::Bounded;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::Direction;
use crate::ArrayRef;
use crate::IntoArray;
use crate::aggregate_fn::accumulator::Accumulator;
use crate::arrays::PrimitiveArray;
use crate::canonical::ToCanonical;
use crate::dtype::NativePType;
use crate::scalar::PValue;
use crate::scalar::Scalar;

pub(super) struct PrimitiveExtremumAccumulator<T, D> {
    current: Option<T>,
    results: Vec<Option<T>>,
    _direction: PhantomData<D>,
}

impl<T: NativePType + Bounded, D: Direction> PrimitiveExtremumAccumulator<T, D>
where
    PValue: From<T>,
{
    pub(super) fn new() -> Self {
        Self {
            current: None,
            results: Vec::new(),
            _direction: PhantomData,
        }
    }

    #[inline]
    fn consider(&mut self, candidate: T) {
        match self.current {
            None => self.current = Some(candidate),
            Some(cur) => {
                if D::should_replace(cur, candidate) {
                    self.current = Some(candidate);
                }
            }
        }
    }
}

fn accumulate_all_valid<T: NativePType + Bounded, D: Direction>(
    values: &[T],
    acc: &mut PrimitiveExtremumAccumulator<T, D>,
) where
    PValue: From<T>,
{
    for &v in values {
        if !v.is_nan() {
            acc.consider(v);
        }
    }
}

fn accumulate_with_mask<T: NativePType + Bounded, D: Direction>(
    values: &[T],
    mask: &vortex_mask::MaskValues,
    acc: &mut PrimitiveExtremumAccumulator<T, D>,
) where
    PValue: From<T>,
{
    for (&v, valid) in values.iter().zip(mask.bit_buffer().iter()) {
        if valid && !v.is_nan() {
            acc.consider(v);
        }
    }
}

impl<T: NativePType + Bounded, D: Direction> Accumulator for PrimitiveExtremumAccumulator<T, D>
where
    PValue: From<T>,
{
    fn accumulate(&mut self, batch: &ArrayRef) -> VortexResult<()> {
        let primitive = batch.to_primitive();
        let validity = primitive.validity_mask()?;
        let values = primitive.as_slice::<T>();

        match &validity {
            Mask::AllTrue(_) => accumulate_all_valid(values, self),
            Mask::AllFalse(_) => {}
            Mask::Values(v) => accumulate_with_mask(values, v, self),
        }

        Ok(())
    }

    fn merge(&mut self, state: &Scalar) -> VortexResult<()> {
        if state.is_null() {
            return Ok(());
        }
        if let Some(v) = state.as_primitive().typed_value::<T>()
            && !v.is_nan()
        {
            self.consider(v);
        }
        Ok(())
    }

    fn is_saturated(&self) -> bool {
        self.current.is_some_and(D::is_saturated)
    }

    fn flush(&mut self) -> VortexResult<()> {
        self.results.push(self.current);
        self.current = None;
        Ok(())
    }

    fn finish(self: Box<Self>) -> VortexResult<ArrayRef> {
        Ok(PrimitiveArray::from_option_iter(self.results).into_array())
    }
}
