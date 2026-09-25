// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::hash::Hasher;

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::VTable;
use vortex_array::vtable::validity_to_child;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::DeltaData;
use crate::FL_CHUNK_SIZE;
use crate::delta::array::DeltaArrayExt;
use crate::delta::array::DeltaArraySlotsExt;
use crate::delta::array::DeltaSlots;
use crate::delta::array::DeltaSlotsView;
use crate::delta::array::delta_decompress::delta_decompress;
use crate::delta::array::lane_count;
use crate::delta_compress;

mod operations;
mod rules;
mod slice;
mod validity;

/// A [`Delta`]-encoded Vortex array.
pub type DeltaArray = Array<Delta>;

#[derive(Clone, prost::Message)]
#[repr(C)]
pub struct DeltaMetadata {
    #[prost(uint64, tag = "1")]
    deltas_len: u64,
    #[prost(uint32, tag = "2")]
    offset: u32, // must be <1024
}

impl ArrayHash for DeltaData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.offset.hash(state);
    }
}

impl ArrayEq for DeltaData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.offset == other.offset
    }
}

impl VTable for Delta {
    type TypedArrayData = DeltaData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("fastlanes.delta");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let delta_slots = DeltaSlotsView::from_slots(slots);
        validate_parts(
            delta_slots.bases,
            delta_slots.deltas,
            delta_slots.validity_child,
            data.offset,
            dtype,
            len,
        )
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("DeltaArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_array::vtable::with_empty_buffers(self, array, buffers)
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        rules::RULES.evaluate(array, parent, child_idx)
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        DeltaSlots::NAMES[idx].to_string()
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            DeltaMetadata {
                deltas_len: array.deltas().len() as u64,
                offset: array.offset() as u32,
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
        vortex_ensure!(
            buffers.is_empty(),
            "DeltaArray expects 0 buffers, got {}",
            buffers.len()
        );
        vortex_ensure!(
            children.len() == 2 || children.len() == 3,
            "DeltaArray expects 2 or 3 children, got {}",
            children.len()
        );
        let metadata = DeltaMetadata::decode(metadata)?;
        let ptype = PType::try_from(dtype)?;
        let lanes = lane_count(ptype);

        let deltas_len = usize::try_from(metadata.deltas_len)
            .map_err(|_| vortex_err!("deltas_len {} overflowed usize", metadata.deltas_len))?;
        vortex_ensure!(
            deltas_len.is_multiple_of(FL_CHUNK_SIZE),
            "deltas length must be a multiple of {FL_CHUNK_SIZE}"
        );
        let bases_len = deltas_len / FL_CHUNK_SIZE * lanes;

        let bases = children.get(0, &dtype.as_nonnullable(), bases_len)?;
        let deltas = children.get(1, &dtype.as_nonnullable(), deltas_len)?;
        let validity_child = if children.len() == 3 {
            Some(children.get(2, &Validity::DTYPE, len)?)
        } else {
            None
        };

        let data = DeltaData::try_new(metadata.offset as usize)?;
        let slots = DeltaSlots {
            bases,
            deltas,
            validity_child,
        }
        .into_slots();
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            delta_decompress(&array, ctx)?.into_array(),
        ))
    }
}

#[derive(Clone, Debug)]
pub struct Delta;

impl Delta {
    /// Construct Delta from nonnullable numeric children and logical validity.
    pub fn try_new(
        bases: ArrayRef,
        deltas: ArrayRef,
        validity: Validity,
        offset: usize,
        len: usize,
    ) -> VortexResult<DeltaArray> {
        let dtype = bases.dtype().with_nullability(validity.nullability());
        let data = DeltaData::try_new(offset)?;
        let slots = DeltaSlots {
            bases,
            deltas,
            validity_child: validity_to_child(&validity, len),
        }
        .into_slots();
        Array::try_from_parts(ArrayParts::new(Delta, dtype, len, data).with_slots(slots))
    }

    /// Compress a primitive array using Delta encoding.
    pub fn try_from_primitive_array(
        array: &PrimitiveArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<DeltaArray> {
        let logical_len = array.len();
        let (bases, deltas) = delta_compress(array, ctx)?;
        Self::try_new(
            bases.into_array(),
            deltas.into_array(),
            array.validity()?,
            0,
            logical_len,
        )
    }
}

fn validate_parts(
    bases: &ArrayRef,
    deltas: &ArrayRef,
    validity_child: Option<&ArrayRef>,
    offset: usize,
    dtype: &DType,
    len: usize,
) -> VortexResult<()> {
    vortex_ensure!(
        offset <= deltas.len() && len <= deltas.len() - offset,
        "offset + len, {offset} + {len}, must be less than or equal to the size of deltas: {}",
        deltas.len()
    );
    vortex_ensure!(
        !bases.dtype().is_nullable() && !deltas.dtype().is_nullable(),
        "DeltaArray: bases and deltas must be nonnullable"
    );
    vortex_ensure!(
        bases.dtype() == deltas.dtype(),
        "DeltaArray: bases and deltas must have the same dtype, got {} and {}",
        bases.dtype(),
        deltas.dtype()
    );

    vortex_ensure!(
        bases.dtype().is_int(),
        "DeltaArray: dtype must be an integer, got {}",
        bases.dtype()
    );

    let expected_dtype = bases.dtype().with_nullability(dtype.nullability());
    vortex_ensure!(
        dtype == &expected_dtype,
        "DeltaArray dtype mismatch: expected {expected_dtype}, got {dtype}"
    );
    if let Some(validity) = validity_child {
        vortex_ensure!(
            dtype.is_nullable(),
            "DeltaArray: validity requires a nullable dtype"
        );
        vortex_ensure!(
            validity.dtype() == &Validity::DTYPE,
            "DeltaArray: validity must be nonnullable bool"
        );
        vortex_ensure!(
            validity.len() == len,
            "DeltaArray: validity length must equal logical length {len}"
        );
    }

    let lanes = lane_count(bases.dtype().as_ptype());

    vortex_ensure!(
        deltas.len().is_multiple_of(1024),
        "deltas length ({}) must be a multiple of 1024",
        deltas.len(),
    );
    vortex_ensure!(
        bases.len() == deltas.len() / FL_CHUNK_SIZE * lanes,
        "bases length ({}) must equal the number of chunks times LANES ({lanes})",
        bases.len(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use prost::Message;
    use vortex_array::test_harness::check_metadata;

    use super::DeltaMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_delta_metadata() {
        check_metadata(
            "delta.metadata",
            &DeltaMetadata {
                offset: u32::MAX,
                deltas_len: u64::MAX,
            }
            .encode_to_vec(),
        );
    }
}
