// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayParts;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::EmptyArrayData;
use crate::array::OperationsVTable;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::array::with_empty_buffers;
use crate::arrays::PrimitiveArray;
use crate::arrays::piecewise_sequence::array::PiecewiseSequenceSlots;
use crate::arrays::piecewise_sequence::check_index_arrays;
use crate::arrays::piecewise_sequence::execute_index_arrays;
use crate::arrays::piecewise_sequence::materialize_ranges;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::dtype::UnsignedPType;
use crate::match_each_unsigned_integer_ptype;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

/// A [`PiecewiseSequence`]-encoded Vortex index array.
pub type PiecewiseSequenceArray = Array<PiecewiseSequence>;

#[derive(Clone, Debug)]
pub struct PiecewiseSequence;

impl VTable for PiecewiseSequence {
    type TypedArrayData = EmptyArrayData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.piecewise-sequence");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        dtype: &DType,
        _len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            dtype == &DType::from(PType::U64),
            "PiecewiseSequenceArray dtype must be u64, got {dtype}"
        );
        vortex_ensure!(
            slots.len() == PiecewiseSequenceSlots::NAMES.len(),
            "PiecewiseSequenceArray requires {} slots, got {}",
            PiecewiseSequenceSlots::NAMES.len(),
            slots.len()
        );
        let starts = slots[PiecewiseSequenceSlots::STARTS]
            .as_ref()
            .ok_or_else(|| vortex_err!("PiecewiseSequenceArray starts slot must be present"))?;
        let lengths = slots[PiecewiseSequenceSlots::LENGTHS]
            .as_ref()
            .ok_or_else(|| vortex_err!("PiecewiseSequenceArray lengths slot must be present"))?;
        let multipliers = slots[PiecewiseSequenceSlots::MULTIPLIERS]
            .as_ref()
            .ok_or_else(|| {
                vortex_err!("PiecewiseSequenceArray multipliers slot must be present")
            })?;
        check_index_arrays(starts, lengths, multipliers)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, _idx: usize) -> BufferHandle {
        vortex_panic!("PiecewiseSequenceArray has no buffers")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        PiecewiseSequenceSlots::NAMES[idx].to_string()
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_bail!("PiecewiseSequenceArray is not serializable")
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let (starts, lengths, multipliers) = execute_index_arrays(array.as_view(), ctx)?;

        let values = match_each_unsigned_integer_ptype!(starts.ptype(), |S| {
            match_each_unsigned_integer_ptype!(lengths.ptype(), |L| {
                match_each_unsigned_integer_ptype!(multipliers.ptype(), |M| {
                    materialize_ranges::<S, L, M>(&starts, &lengths, &multipliers, array.len())?
                })
            })
        });
        Ok(ExecutionResult::done(
            PrimitiveArray::new(values.freeze(), Validity::NonNullable).into_array(),
        ))
    }
}

impl OperationsVTable<PiecewiseSequence> for PiecewiseSequence {
    fn scalar_at(
        array: ArrayView<'_, PiecewiseSequence>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let (starts, lengths, multipliers) = execute_index_arrays(array, ctx)?;

        let value = match_each_unsigned_integer_ptype!(starts.ptype(), |S| {
            match_each_unsigned_integer_ptype!(lengths.ptype(), |L| {
                match_each_unsigned_integer_ptype!(multipliers.ptype(), |M| {
                    scalar_at::<S, L, M>(&starts, &lengths, &multipliers, index)?
                })
            })
        });
        Ok(value.into())
    }
}

impl ValidityVTable<PiecewiseSequence> for PiecewiseSequence {
    fn validity(_array: ArrayView<'_, PiecewiseSequence>) -> VortexResult<Validity> {
        Ok(Validity::NonNullable)
    }
}

fn scalar_at<S, L, M>(
    starts: &PrimitiveArray,
    lengths: &PrimitiveArray,
    multipliers: &PrimitiveArray,
    index: usize,
) -> VortexResult<u64>
where
    S: UnsignedPType,
    L: UnsignedPType,
    M: UnsignedPType,
{
    let mut remaining = index;
    for ((&start, &length), &multiplier) in starts
        .as_slice::<S>()
        .iter()
        .zip_eq(lengths.as_slice::<L>())
        .zip_eq(multipliers.as_slice::<M>())
    {
        let length: usize = length.as_();
        if remaining < length {
            let start: usize = start.as_();
            let multiplier: usize = multiplier.as_();
            let offset = remaining
                .checked_mul(multiplier)
                .ok_or_else(|| vortex_err!("PiecewiseSequenceArray range overflows usize"))?;
            let value = start
                .checked_add(offset)
                .ok_or_else(|| vortex_err!("PiecewiseSequenceArray range overflows usize"))?;
            return Ok(value as u64);
        }
        remaining -= length;
    }
    vortex_bail!("PiecewiseSequenceArray index {index} out of bounds")
}
