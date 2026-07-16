// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use prost::Message;
use smallvec::smallvec;
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
use crate::arrays::piecewise_sequence::array::PiecewiseSequenceArraySlotsExt;
use crate::arrays::piecewise_sequence::array::PiecewiseSequenceSlots;
use crate::arrays::piecewise_sequence::check_index_arrays;
use crate::arrays::piecewise_sequence::execute_index_arrays;
use crate::arrays::piecewise_sequence::materialize_ranges;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::match_each_unsigned_integer_ptype;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

/// A [`PiecewiseSequence`]-encoded Vortex index array.
pub type PiecewiseSequenceArray = Array<PiecewiseSequence>;

#[derive(Clone, prost::Message)]
struct PiecewiseSequenceMetadata {
    #[prost(uint64, tag = "1")]
    num_pieces: u64,
    #[prost(enumeration = "PType", tag = "2")]
    starts_ptype: i32,
    #[prost(enumeration = "PType", tag = "3")]
    lengths_ptype: i32,
    #[prost(enumeration = "PType", tag = "4")]
    multipliers_ptype: i32,
}

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
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            PiecewiseSequenceMetadata {
                num_pieces: u64::try_from(array.starts().len()).map_err(|_| {
                    vortex_err!(
                        "PiecewiseSequenceArray piece count {} overflowed u64",
                        array.starts().len()
                    )
                })?,
                starts_ptype: PType::try_from(array.starts().dtype())? as i32,
                lengths_ptype: PType::try_from(array.lengths().dtype())? as i32,
                multipliers_ptype: PType::try_from(array.multipliers().dtype())? as i32,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = PiecewiseSequenceMetadata::decode(metadata)?;
        vortex_ensure!(
            dtype == &DType::from(PType::U64),
            "PiecewiseSequenceArray dtype must be u64, got {dtype}"
        );
        vortex_ensure!(
            buffers.is_empty(),
            "PiecewiseSequenceArray expects no buffers, got {}",
            buffers.len()
        );
        vortex_ensure!(
            children.len() == PiecewiseSequenceSlots::NAMES.len(),
            "PiecewiseSequenceArray expects {} children, got {}",
            PiecewiseSequenceSlots::NAMES.len(),
            children.len()
        );

        let num_pieces = usize::try_from(metadata.num_pieces).map_err(|_| {
            vortex_err!(
                "PiecewiseSequenceArray piece count {} does not fit in usize",
                metadata.num_pieces
            )
        })?;
        let starts = children.get(
            PiecewiseSequenceSlots::STARTS,
            &metadata.starts_ptype().into(),
            num_pieces,
        )?;
        let lengths = children.get(
            PiecewiseSequenceSlots::LENGTHS,
            &metadata.lengths_ptype().into(),
            num_pieces,
        )?;
        let multipliers = children.get(
            PiecewiseSequenceSlots::MULTIPLIERS,
            &metadata.multipliers_ptype().into(),
            num_pieces,
        )?;

        Ok(
            ArrayParts::new(self.clone(), dtype.clone(), len, EmptyArrayData)
                .with_slots(smallvec![Some(starts), Some(lengths), Some(multipliers)]),
        )
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
    S: crate::dtype::UnsignedPType,
    L: crate::dtype::UnsignedPType,
    M: crate::dtype::UnsignedPType,
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
