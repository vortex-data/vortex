// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernels;
mod operations;
mod slice;

use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::Canonical;
use crate::DeserializeMetadata;
use crate::DynArray;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::Precision;
use crate::ProstMetadata;
use crate::arrays::PrimitiveArray;
use crate::arrays::patched::PatchedArray;
use crate::arrays::patched::compute::rules::PARENT_RULES;
use crate::arrays::patched::patch_lanes;
use crate::arrays::patched::vtable::kernels::PARENT_KERNELS;
use crate::arrays::primitive::PrimitiveArrayParts;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;
use crate::serde::ArrayChildren;
use crate::stats::ArrayStats;
use crate::stats::StatsSetRef;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::VTable;
use crate::vtable::ValidityChild;
use crate::vtable::ValidityVTableFromChild;

vtable!(Patched);

#[derive(Clone, Debug)]
pub struct Patched;

impl ValidityChild<Patched> for Patched {
    fn validity_child(array: &PatchedArray) -> &ArrayRef {
        &array.inner
    }
}

#[derive(Clone, prost::Message)]
pub struct PatchedMetadata {
    #[prost(uint32, tag = "1")]
    pub(crate) offset: u32,
}

impl VTable for Patched {
    type Array = PatchedArray;
    type Metadata = ProstMetadata<PatchedMetadata>;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn vtable(_array: &Self::Array) -> &Self {
        &Patched
    }

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("vortex.patched")
    }

    fn len(array: &Self::Array) -> usize {
        array.len
    }

    fn dtype(array: &Self::Array) -> &DType {
        array.inner.dtype()
    }

    fn stats(array: &Self::Array) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: Hasher>(array: &Self::Array, state: &mut H, precision: Precision) {
        array.inner.array_hash(state, precision);
        array.values_ptype.hash(state);
        array.n_chunks.hash(state);
        array.n_lanes.hash(state);
        array.lane_offsets.array_hash(state, precision);
        array.indices.array_hash(state, precision);
        array.values.array_hash(state, precision);
    }

    fn array_eq(array: &Self::Array, other: &Self::Array, precision: Precision) -> bool {
        array.n_chunks == other.n_chunks
            && array.n_lanes == other.n_lanes
            && array.values_ptype == other.values_ptype
            && array.inner.array_eq(&other.inner, precision)
            && array.lane_offsets.array_eq(&other.lane_offsets, precision)
            && array.indices.array_eq(&other.indices, precision)
            && array.values.array_eq(&other.values, precision)
    }

    fn nbuffers(_array: &Self::Array) -> usize {
        3
    }

    fn buffer(array: &Self::Array, idx: usize) -> BufferHandle {
        match idx {
            0 => array.lane_offsets.clone(),
            1 => array.indices.clone(),
            2 => array.values.clone(),
            _ => vortex_panic!("invalid buffer index for PatchedArray: {idx}"),
        }
    }

    fn buffer_name(_array: &Self::Array, idx: usize) -> Option<String> {
        match idx {
            0 => Some("lane_offsets".to_string()),
            1 => Some("patch_indices".to_string()),
            2 => Some("patch_values".to_string()),
            _ => vortex_panic!("invalid buffer index for PatchedArray: {idx}"),
        }
    }

    fn nchildren(_array: &Self::Array) -> usize {
        1
    }

    fn child(array: &Self::Array, idx: usize) -> ArrayRef {
        if idx == 0 {
            array.inner.clone()
        } else {
            vortex_panic!("invalid child index for PatchedArray: {idx}");
        }
    }

    fn child_name(_array: &Self::Array, idx: usize) -> String {
        if idx == 0 {
            "inner".to_string()
        } else {
            vortex_panic!("invalid child index for PatchedArray: {idx}");
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn metadata(array: &Self::Array) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(PatchedMetadata {
            offset: array.offset as u32,
        }))
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        let inner = <ProstMetadata<PatchedMetadata> as DeserializeMetadata>::deserialize(bytes)?;
        Ok(ProstMetadata(inner))
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<PatchedArray> {
        let inner = children.get(0, dtype, len)?;

        let n_chunks = len.div_ceil(1024);

        let n_lanes = match_each_native_ptype!(dtype.as_ptype(), |P| { patch_lanes::<P>() });

        let &[lane_offsets, indices, values] = &buffers else {
            vortex_bail!("invalid buffer count for PatchedArray");
        };

        Ok(PatchedArray {
            inner,
            n_chunks,
            n_lanes,
            offset: metadata.offset as usize,
            len,
            lane_offsets: lane_offsets.clone(),
            indices: indices.clone(),
            values: values.clone(),
            values_ptype: dtype.as_ptype(),
            stats_set: ArrayStats::default(),
        })
    }

    fn with_children(array: &mut Self::Array, mut children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 1,
            "PatchedArray must have exactly 1 child"
        );

        array.inner = children.remove(0);

        Ok(())
    }

    fn execute(array: Arc<Self::Array>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let inner = array
            .inner
            .clone()
            .execute::<Canonical>(ctx)?
            .into_primitive();

        let PrimitiveArrayParts {
            buffer,
            ptype,
            validity,
        } = inner.into_parts();

        let lane_offsets: Buffer<u32> =
            Buffer::from_byte_buffer(array.lane_offsets.clone().unwrap_host());
        let indices: Buffer<u16> = Buffer::from_byte_buffer(array.indices.clone().unwrap_host());

        let patched_values = match_each_native_ptype!(array.values_ptype, |V| {
            let mut output = Buffer::<V>::from_byte_buffer(buffer.unwrap_host()).into_mut();
            let values: Buffer<V> = Buffer::from_byte_buffer(array.values.clone().unwrap_host());

            let offset = array.offset;
            let len = array.len;

            apply::<V>(
                &mut output,
                offset,
                len,
                array.n_chunks,
                array.n_lanes,
                &lane_offsets,
                &indices,
                &values,
            );

            // The output will always be aligned to a chunk boundary, we apply the offset/len
            // at the end to slice to only the in-bounds values.
            let _output = output.as_slice();
            let output = output.freeze().slice(offset..offset + len);

            PrimitiveArray::from_byte_buffer(output.into_byte_buffer(), ptype, validity)
        });

        Ok(ExecutionResult::done(patched_values.into_array()))
    }

    fn execute_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}

/// Apply patches on top of the existing value types.
#[allow(clippy::too_many_arguments)]
fn apply<V: NativePType>(
    output: &mut [V],
    offset: usize,
    len: usize,
    n_chunks: usize,
    n_lanes: usize,
    lane_offsets: &[u32],
    indices: &[u16],
    values: &[V],
) {
    for chunk in 0..n_chunks {
        let start = lane_offsets[chunk * n_lanes] as usize;
        let stop = lane_offsets[chunk * n_lanes + n_lanes] as usize;

        for idx in start..stop {
            // the indices slice is measured as an offset into the 1024-value chunk.
            let index = chunk * 1024 + indices[idx] as usize;
            if index < offset || index >= offset + len {
                continue;
            }

            let value = values[idx];
            output[index] = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_buffer::buffer_mut;
    use vortex_session::VortexSession;

    use crate::Canonical;
    use crate::ExecutionCtx;
    use crate::IntoArray;
    use crate::arrays::PatchedArray;
    use crate::dtype::Nullability;
    use crate::patches::Patches;
    use crate::scalar::Scalar;

    #[test]
    fn test_execute() {
        let values = buffer![0u16; 1024].into_array();
        let patches = Patches::new(
            1024,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![1u16; 3].into_array(),
            None,
        )
        .unwrap();

        let session = VortexSession::empty();
        let mut ctx = ExecutionCtx::new(session);

        let array = PatchedArray::from_array_and_patches(values, &patches, &mut ctx)
            .unwrap()
            .into_array();

        let executed = array
            .execute::<Canonical>(&mut ctx)
            .unwrap()
            .into_primitive()
            .into_buffer::<u16>();

        let mut expected = buffer_mut![0u16; 1024];
        expected[1] = 1;
        expected[2] = 1;
        expected[3] = 1;

        assert_eq!(executed, expected.freeze());
    }

    #[test]
    fn test_scalar_at() {
        let values = buffer![0u16; 1024].into_array();
        let patches = Patches::new(
            1024,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![1u16; 3].into_array(),
            None,
        )
        .unwrap();

        let session = VortexSession::empty();
        let mut ctx = ExecutionCtx::new(session);

        let array = PatchedArray::from_array_and_patches(values, &patches, &mut ctx)
            .unwrap()
            .into_array();

        assert_eq!(
            array.scalar_at(0).unwrap(),
            Scalar::primitive(0u16, Nullability::NonNullable)
        );
        assert_eq!(
            array.scalar_at(1).unwrap(),
            Scalar::primitive(1u16, Nullability::NonNullable)
        );
        assert_eq!(
            array.scalar_at(2).unwrap(),
            Scalar::primitive(1u16, Nullability::NonNullable)
        );
        assert_eq!(
            array.scalar_at(3).unwrap(),
            Scalar::primitive(1u16, Nullability::NonNullable)
        );
    }
}
