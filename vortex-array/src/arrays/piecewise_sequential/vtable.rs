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
use crate::arrays::piecewise_sequential::array::PiecewiseSequentialArraySlotsExt;
use crate::arrays::piecewise_sequential::array::PiecewiseSequentialSlots;
use crate::arrays::piecewise_sequential::check_index_arrays;
use crate::arrays::piecewise_sequential::index_value_to_u64;
use crate::arrays::piecewise_sequential::index_value_to_usize;
use crate::arrays::piecewise_sequential::materialize_ranges;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::match_each_unsigned_integer_ptype;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

/// A [`PiecewiseSequential`]-encoded Vortex index array.
pub type PiecewiseSequentialArray = Array<PiecewiseSequential>;

#[derive(Clone, prost::Message)]
struct PiecewiseSequentialMetadata {
    #[prost(uint64, tag = "1")]
    num_pieces: u64,
    #[prost(enumeration = "PType", tag = "2")]
    starts_ptype: i32,
    #[prost(enumeration = "PType", tag = "3")]
    lengths_ptype: i32,
}

#[derive(Clone, Debug)]
pub struct PiecewiseSequential;

impl VTable for PiecewiseSequential {
    type TypedArrayData = EmptyArrayData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.piecewise-sequential");
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
            "PiecewiseSequentialArray dtype must be u64, got {dtype}"
        );
        vortex_ensure!(
            slots.len() == PiecewiseSequentialSlots::NAMES.len(),
            "PiecewiseSequentialArray requires {} slots, got {}",
            PiecewiseSequentialSlots::NAMES.len(),
            slots.len()
        );
        let starts = slots[PiecewiseSequentialSlots::STARTS]
            .as_ref()
            .ok_or_else(|| {
                vortex_error::vortex_err!("PiecewiseSequentialArray starts slot must be present")
            })?;
        let lengths = slots[PiecewiseSequentialSlots::LENGTHS]
            .as_ref()
            .ok_or_else(|| {
                vortex_error::vortex_err!("PiecewiseSequentialArray lengths slot must be present")
            })?;
        check_index_arrays(starts, lengths)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, _idx: usize) -> BufferHandle {
        vortex_panic!("PiecewiseSequentialArray has no buffers")
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
        PiecewiseSequentialSlots::NAMES[idx].to_string()
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            PiecewiseSequentialMetadata {
                num_pieces: u64::try_from(array.starts().len()).map_err(|_| {
                    vortex_err!(
                        "PiecewiseSequentialArray piece count {} overflowed u64",
                        array.starts().len()
                    )
                })?,
                starts_ptype: PType::try_from(array.starts().dtype())? as i32,
                lengths_ptype: PType::try_from(array.lengths().dtype())? as i32,
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
        let metadata = PiecewiseSequentialMetadata::decode(metadata)?;
        vortex_ensure!(
            dtype == &DType::from(PType::U64),
            "PiecewiseSequentialArray dtype must be u64, got {dtype}"
        );
        vortex_ensure!(
            buffers.is_empty(),
            "PiecewiseSequentialArray expects no buffers, got {}",
            buffers.len()
        );
        vortex_ensure!(
            children.len() == PiecewiseSequentialSlots::NAMES.len(),
            "PiecewiseSequentialArray expects {} children, got {}",
            PiecewiseSequentialSlots::NAMES.len(),
            children.len()
        );

        let num_pieces = usize::try_from(metadata.num_pieces).map_err(|_| {
            vortex_err!(
                "PiecewiseSequentialArray piece count {} does not fit in usize",
                metadata.num_pieces
            )
        })?;
        let starts = children.get(
            PiecewiseSequentialSlots::STARTS,
            &metadata.starts_ptype().into(),
            num_pieces,
        )?;
        let lengths = children.get(
            PiecewiseSequentialSlots::LENGTHS,
            &metadata.lengths_ptype().into(),
            num_pieces,
        )?;

        Ok(
            ArrayParts::new(self.clone(), dtype.clone(), len, EmptyArrayData)
                .with_slots(smallvec![Some(starts), Some(lengths)]),
        )
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let starts = array.starts().clone().execute::<PrimitiveArray>(ctx)?;
        let lengths = array.lengths().clone().execute::<PrimitiveArray>(ctx)?;
        check_index_arrays(starts.as_ref(), lengths.as_ref())?;

        let values = match_each_unsigned_integer_ptype!(starts.ptype(), |S| {
            match_each_unsigned_integer_ptype!(lengths.ptype(), |L| {
                materialize_ranges::<S, L>(&starts, &lengths, array.len())?
            })
        });
        Ok(ExecutionResult::done(
            PrimitiveArray::new(values.freeze(), Validity::NonNullable).into_array(),
        ))
    }
}

impl OperationsVTable<PiecewiseSequential> for PiecewiseSequential {
    fn scalar_at(
        array: ArrayView<'_, PiecewiseSequential>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let starts = array.starts().clone().execute::<PrimitiveArray>(ctx)?;
        let lengths = array.lengths().clone().execute::<PrimitiveArray>(ctx)?;
        check_index_arrays(starts.as_ref(), lengths.as_ref())?;

        let value = match_each_unsigned_integer_ptype!(starts.ptype(), |S| {
            match_each_unsigned_integer_ptype!(lengths.ptype(), |L| {
                scalar_at::<S, L>(&starts, &lengths, index)?
            })
        });
        Ok(value.into())
    }
}

impl ValidityVTable<PiecewiseSequential> for PiecewiseSequential {
    fn validity(_array: ArrayView<'_, PiecewiseSequential>) -> VortexResult<Validity> {
        Ok(Validity::NonNullable)
    }
}

fn scalar_at<S, L>(
    starts: &PrimitiveArray,
    lengths: &PrimitiveArray,
    index: usize,
) -> VortexResult<u64>
where
    S: crate::dtype::UnsignedPType + num_traits::AsPrimitive<u64>,
    L: crate::dtype::UnsignedPType,
{
    let mut remaining = index;
    for (&start, &length) in starts
        .as_slice::<S>()
        .iter()
        .zip_eq(lengths.as_slice::<L>())
    {
        let length = index_value_to_usize(length);
        if remaining < length {
            return Ok(index_value_to_u64(start) + remaining as u64);
        }
        remaining -= length;
    }
    vortex_bail!("PiecewiseSequentialArray index {index} out of bounds")
}
