// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use fastlanes::FastLanes;
use prost::Message;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::StatsSetRef;
use vortex_array::vtable;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTableFromChildSliceHelper;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::DeltaArray;
use crate::DeltaArrayExt;
use crate::delta::array::delta_decompress::delta_decompress;

mod operations;
mod rules;
mod slice;
mod validity;

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

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChildSliceHelper;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
    }

    fn len(array: &DeltaArray) -> usize {
        array.len()
    }

    fn dtype(array: &DeltaArray) -> &DType {
        array.dtype()
    }

    fn stats(array: &DeltaArray) -> StatsSetRef<'_> {
        array.stats_set().to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &DeltaArray, state: &mut H, precision: Precision) {
        array.offset().hash(state);
        array.len().hash(state);
        array.dtype().hash(state);
        array.bases().array_hash(state, precision);
        array.deltas().array_hash(state, precision);
    }

    fn array_eq(array: &DeltaArray, other: &DeltaArray, precision: Precision) -> bool {
        array.offset() == other.offset()
            && array.len() == other.len()
            && array.dtype() == other.dtype()
            && array.bases().array_eq(other.bases(), precision)
            && array.deltas().array_eq(other.deltas(), precision)
    }

    fn nbuffers(_array: &DeltaArray) -> usize {
        0
    }

    fn buffer(_array: &DeltaArray, idx: usize) -> BufferHandle {
        vortex_panic!("DeltaArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &DeltaArray, _idx: usize) -> Option<String> {
        None
    }

    fn nchildren(_array: &DeltaArray) -> usize {
        2
    }

    fn child(array: &DeltaArray, idx: usize) -> ArrayRef {
        match idx {
            0 => array.bases().clone(),
            1 => array.deltas().clone(),
            _ => vortex_panic!("DeltaArray child index {idx} out of bounds"),
        }
    }

    fn child_name(_array: &DeltaArray, idx: usize) -> String {
        match idx {
            0 => "bases".to_string(),
            1 => "deltas".to_string(),
            _ => vortex_panic!("DeltaArray child name index {idx} out of bounds"),
        }
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        rules::RULES.evaluate(array, parent, child_idx)
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

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(DeltaMetadata::decode(bytes)?))
    }

    fn build(
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

        Self::try_new(bases, deltas, metadata.0.offset as usize, len)
    }

    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        Ok(delta_decompress(array, ctx)?.into_array())
    }
}

#[derive(Debug)]
pub struct DeltaVTable;

impl DeltaVTable {
    pub const ID: ArrayId = ArrayId::new_ref("fastlanes.delta");

    // TODO(ngates): remove constructing from vec
    pub fn try_from_vec<T: vortex_array::dtype::NativePType>(
        vec: Vec<T>,
    ) -> VortexResult<DeltaArray> {
        use vortex_array::arrays::PrimitiveArray;
        use vortex_array::validity::Validity;
        use vortex_buffer::Buffer;

        Self::try_from_primitive_array(&PrimitiveArray::new(
            Buffer::copy_from(vec),
            Validity::NonNullable,
        ))
    }

    pub fn try_from_primitive_array(
        array: &vortex_array::arrays::PrimitiveArray,
    ) -> VortexResult<DeltaArray> {
        use vortex_array::IntoArray;

        use crate::delta::array::delta_compress;

        let (bases, deltas) = delta_compress::delta_compress(array)?;

        Self::try_from_delta_compress_parts(bases.into_array(), deltas.into_array())
    }

    /// Create a [`DeltaArray`] from the given `bases` and `deltas` arrays.
    /// Note the `deltas` might be nullable
    pub fn try_from_delta_compress_parts(
        bases: ArrayRef,
        deltas: ArrayRef,
    ) -> VortexResult<DeltaArray> {
        let logical_len = deltas.len();
        Self::try_new(bases, deltas, 0, logical_len)
    }

    pub fn try_new(
        bases: ArrayRef,
        deltas: ArrayRef,
        offset: usize,
        logical_len: usize,
    ) -> VortexResult<DeltaArray> {
        use vortex_array::dtype::DType;
        use vortex_error::vortex_bail;

        use crate::delta::array::lane_count;

        if offset >= 1024 {
            vortex_bail!("offset must be less than 1024: {}", offset);
        }
        if offset + logical_len > deltas.len() {
            vortex_bail!(
                "offset + logical_len, {} + {}, must be less than or equal to the size of deltas: {}",
                offset,
                logical_len,
                deltas.len()
            )
        }
        if !bases.dtype().eq_ignore_nullability(deltas.dtype()) {
            vortex_bail!(
                "DeltaArray: bases and deltas must have the same dtype, got {:?} and {:?}",
                bases.dtype(),
                deltas.dtype()
            );
        }
        let DType::Primitive(ptype, _) = bases.dtype().clone() else {
            vortex_bail!(
                "DeltaArray: dtype must be an integer, got {}",
                bases.dtype()
            );
        };

        if !ptype.is_int() {
            vortex_bail!("DeltaArray: ptype must be an integer, got {}", ptype);
        }

        let lanes = lane_count(ptype);

        if deltas.len().is_multiple_of(1024) != bases.len().is_multiple_of(lanes) {
            vortex_bail!(
                "deltas length ({}) is a multiple of 1024 iff bases length ({}) is a multiple of LANES ({})",
                deltas.len(),
                bases.len(),
                lanes,
            );
        }

        // SAFETY: validation done above
        Ok(unsafe { DeltaArray::new_unchecked(bases, deltas, offset, logical_len) })
    }
}

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
