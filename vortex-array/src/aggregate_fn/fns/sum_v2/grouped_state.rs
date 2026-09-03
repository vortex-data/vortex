// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::decode_partial_scalar;
use super::sum_v2_partial_dtype;
use super::sum_v2_partial_fields;
use crate::ArrayRef;
use crate::IntoArray;
use crate::aggregate_fn::GroupedState;
use crate::aggregate_fn::fns::sum::DenseSums;
use crate::arrays::BoolArray;
use crate::arrays::StructArray;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// Dense grouped [`SumV2`](super::SumV2) state.
///
/// Each group holds one native sum plus the overflow and empty flags that give `SumV2` its SQL
/// semantics: a group that saw no valid value finalizes to null rather than to zero.
pub(crate) struct SumV2GroupedState {
    sums: DenseSums,
    empty: Vec<bool>,
    sum_dtype: DType,
}

impl SumV2GroupedState {
    pub(crate) fn try_new(sum_dtype: DType) -> VortexResult<Self> {
        Ok(Self {
            sums: DenseSums::try_new(&sum_dtype)?,
            empty: Vec::new(),
            sum_dtype,
        })
    }

    /// The dense sums and per-group empty flags, for use by the grouped kernel.
    pub(crate) fn parts_mut(&mut self) -> (&mut DenseSums, &mut [bool]) {
        (&mut self.sums, &mut self.empty)
    }
}

impl GroupedState for SumV2GroupedState {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn len(&self) -> usize {
        self.sums.len()
    }

    fn ensure_groups(&mut self, num_groups: usize) -> VortexResult<()> {
        self.sums.ensure_groups(num_groups);
        self.empty.resize(self.sums.len(), true);
        Ok(())
    }

    fn is_saturated(&self, group_id: usize) -> bool {
        self.sums.is_saturated(group_id)
    }

    fn combine_scalar(&mut self, group_id: usize, partial: Scalar) -> VortexResult<()> {
        let (sum, is_overflow, is_empty) = decode_partial_scalar(partial)?;
        if is_empty {
            return Ok(());
        }
        self.empty[group_id] = false;
        if is_overflow {
            self.sums.set_overflowed(group_id);
            return Ok(());
        }
        self.sums.add_scalar(group_id, &sum)
    }

    fn partial_scalar(&self, group_id: usize) -> VortexResult<Scalar> {
        Ok(Scalar::struct_(
            sum_v2_partial_dtype(self.sum_dtype.clone()),
            vec![
                self.sums.value_scalar(group_id, Nullability::NonNullable)?,
                Scalar::bool(self.sums.is_overflowed(group_id), Nullability::NonNullable),
                Scalar::bool(
                    self.empty.get(group_id).copied().unwrap_or(true),
                    Nullability::NonNullable,
                ),
            ],
        ))
    }

    fn flush_partials(&mut self) -> VortexResult<ArrayRef> {
        let empty = std::mem::take(&mut self.empty);
        let overflowed = self.sums.take_overflowed();
        let len = empty.len();
        let sums = self.sums.take_values(Validity::NonNullable)?;
        let fields = sum_v2_partial_fields(sums.dtype().clone());
        Ok(StructArray::try_new(
            fields.names().clone(),
            vec![
                sums,
                BoolArray::from_iter(overflowed).into_array(),
                BoolArray::from_iter(empty).into_array(),
            ],
            len,
            Validity::AllValid,
        )
        .map_err(|err| vortex_err!("Failed to build grouped sum_v2 partials: {err}"))?
        .into_array())
    }
}
