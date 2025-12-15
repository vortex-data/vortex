// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::Delta;
use fastlanes::FastLanes;
use fastlanes::Transpose;
use num_traits::WrappingAdd;
use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::ProstMetadata;
use vortex_array::VectorExecutor;
use vortex_array::serde::ArrayChildren;
use vortex_array::vtable;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::ArrayVTable;
use vortex_array::vtable::ArrayVTableExt;
use vortex_array::vtable::NotSupported;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTableFromChildSliceHelper;
use vortex_buffer::BufferHandle;
use vortex_dtype::DType;
use vortex_dtype::NativePType;
use vortex_dtype::PType;
use vortex_dtype::PTypeDowncastExt;
use vortex_dtype::match_each_unsigned_integer_ptype;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_vector::Vector;
use vortex_vector::primitive::PVector;
use vortex_vector::primitive::PrimitiveVector;

use crate::DeltaArray;
use crate::delta::array::delta_decompress::decompress_primitive;

mod array;
mod canonical;
mod operations;
mod validity;
mod visitor;

vtable!(Delta);

#[derive(Clone, prost::Message)]
#[repr(C)]
pub struct DeltaMetadata {
    #[prost(uint64, tag = "1")]
    deltas_len: u64,
    #[prost(uint32, tag = "2")]
    offset: u32, // must be <1024
}

impl VTable for DeltaVTable {
    type Array = DeltaArray;

    type Metadata = ProstMetadata<DeltaMetadata>;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChildSliceHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("fastlanes.delta")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        DeltaVTable.as_vtable()
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        // DeltaArray children order (from visit_children):
        // 1. bases
        // 2. deltas

        vortex_ensure!(
            children.len() == 2,
            "Expected 2 children for Delta encoding, got {}",
            children.len()
        );

        array.bases = children[0].clone();
        array.deltas = children[1].clone();

        Ok(())
    }

    fn metadata(array: &DeltaArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(DeltaMetadata {
            deltas_len: array.deltas().len() as u64,
            offset: array.offset() as u32,
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.0.encode_to_vec()))
    }

    fn deserialize(buffer: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(DeltaMetadata::decode(buffer)?))
    }

    fn build(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<DeltaArray> {
        assert_eq!(children.len(), 2);
        let ptype = PType::try_from(dtype)?;
        let lanes = match_each_unsigned_integer_ptype!(ptype, |T| { <T as FastLanes>::LANES });

        // Compute the length of the bases array
        let deltas_len = usize::try_from(metadata.0.deltas_len)
            .map_err(|_| vortex_err!("deltas_len {} overflowed usize", metadata.0.deltas_len))?;
        let num_chunks = deltas_len / 1024;
        let remainder_base_size = if deltas_len % 1024 > 0 { 1 } else { 0 };
        let bases_len = num_chunks * lanes + remainder_base_size;

        let bases = children.get(0, dtype, bases_len)?;
        let deltas = children.get(1, dtype, deltas_len)?;

        DeltaArray::try_new(bases, deltas, metadata.0.offset as usize, len)
    }

    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<Vector> {
        let bases = array.bases().execute(ctx)?.into_primitive();
        let deltas = array.deltas().execute(ctx)?.into_primitive();

        let start = array.offset();
        let end = start + array.len();
        let validity = array.deltas().validity_mask().slice(start..end);

        Ok(match bases {
            PrimitiveVector::U8(pv) => {
                decompress::<u8, { u8::LANES }>(&pv, &deltas, start, end, validity)
            }
            PrimitiveVector::U16(pv) => {
                decompress::<u16, { u16::LANES }>(&pv, &deltas, start, end, validity)
            }
            PrimitiveVector::U32(pv) => {
                decompress::<u32, { u32::LANES }>(&pv, &deltas, start, end, validity)
            }
            PrimitiveVector::U64(pv) => {
                decompress::<u64, { u64::LANES }>(&pv, &deltas, start, end, validity)
            }
            PrimitiveVector::I8(_)
            | PrimitiveVector::I16(_)
            | PrimitiveVector::I32(_)
            | PrimitiveVector::I64(_)
            | PrimitiveVector::F16(_)
            | PrimitiveVector::F32(_)
            | PrimitiveVector::F64(_) => {
                vortex_panic!("Tried to match a non-unsigned vector in an unsigned match statement")
            }
        })
    }
}

/// Decompresses delta-encoded data for a specific primitive type.
fn decompress<T, const LANES: usize>(
    bases: &PVector<T>,
    deltas: &PrimitiveVector,
    start: usize,
    end: usize,
    validity: Mask,
) -> Vector
where
    T: NativePType + Delta + Transpose + WrappingAdd,
{
    let buffer = decompress_primitive::<T, LANES>(bases.as_ref(), deltas.downcast::<T>().as_ref());
    let buffer = buffer.slice(start..end);

    // SAFETY: We slice the buffer and the validity by the same range.
    unsafe { PVector::<T>::new_unchecked(buffer, validity) }.into()
}

#[derive(Debug)]
pub struct DeltaVTable;

#[cfg(test)]
mod tests {
    use vortex_array::test_harness::check_metadata;

    use super::DeltaMetadata;
    use super::ProstMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_delta_metadata() {
        check_metadata(
            "delta.metadata",
            ProstMetadata(DeltaMetadata {
                offset: u32::MAX,
                deltas_len: u64::MAX,
            }),
        );
    }
}
