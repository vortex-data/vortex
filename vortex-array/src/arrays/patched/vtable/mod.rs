// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernels;
mod operations;
mod slice;

use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
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
use crate::SerializeMetadata;
use crate::arrays::PrimitiveArray;
use crate::arrays::patched::PatchedArray;
use crate::arrays::patched::compute::rules::PARENT_RULES;
use crate::arrays::patched::patch_lanes;
use crate::arrays::patched::vtable::kernels::PARENT_KERNELS;
use crate::arrays::primitive::PrimitiveArrayParts;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::PrimitiveBuilder;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::serde::ArrayChildren;
use crate::stats::ArrayStats;
use crate::stats::StatsSetRef;
use crate::vtable;
use crate::vtable::Array;
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
    /// Length of the `inner` child.
    ///
    /// This may not match the length of the wrapping PatchedArray, if for example
    /// in a filter or slice it may be sliced to the nearest chunk boundary.
    #[prost(uint64, tag = "1")]
    pub(crate) inner_len: u64,

    /// Offset within the first chunk of `inner` where data begins.
    ///
    /// This may become nonzero after slicing.
    #[prost(uint32, tag = "2")]
    pub(crate) offset: u32,

    /// Number of patches. This is the length of the `indices` and `values` children.
    #[prost(uint32, tag = "3")]
    pub(crate) n_patches: u32,

    /// Number of lanes the patches get spread over.
    ///
    /// By default, this is either 16 or 32 depending on the width of the type, but may change
    /// in the future, so we save it on write.
    #[prost(uint32, tag = "4")]
    pub(crate) n_lanes: u32,
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
        array.n_chunks.hash(state);
        array.n_lanes.hash(state);
        array.lane_offsets.array_hash(state, precision);
        array.indices.array_hash(state, precision);
        array.values.array_hash(state, precision);
    }

    fn array_eq(array: &Self::Array, other: &Self::Array, precision: Precision) -> bool {
        array.n_chunks == other.n_chunks
            && array.n_lanes == other.n_lanes
            && array.inner.array_eq(&other.inner, precision)
            && array.lane_offsets.array_eq(&other.lane_offsets, precision)
            && array.indices.array_eq(&other.indices, precision)
            && array.values.array_eq(&other.values, precision)
    }

    fn nbuffers(_array: &Self::Array) -> usize {
        1
    }

    fn buffer(array: &Self::Array, idx: usize) -> BufferHandle {
        match idx {
            0 => array.lane_offsets.clone(),
            _ => vortex_panic!("invalid buffer index for PatchedArray: {idx}"),
        }
    }

    fn buffer_name(_array: &Self::Array, idx: usize) -> Option<String> {
        match idx {
            0 => Some("lane_offsets".to_string()),
            _ => vortex_panic!("invalid buffer index for PatchedArray: {idx}"),
        }
    }

    fn nchildren(_array: &Self::Array) -> usize {
        3
    }

    fn child(array: &Self::Array, idx: usize) -> ArrayRef {
        match idx {
            0 => array.inner.clone(),
            1 => array.indices.clone(),
            2 => array.values.clone(),
            _ => vortex_panic!("invalid child index for PatchedArray: {idx}"),
        }
    }

    fn child_name(_array: &Self::Array, idx: usize) -> String {
        match idx {
            0 => "inner".to_string(),
            1 => "patch_indices".to_string(),
            2 => "patch_values".to_string(),
            _ => vortex_panic!("invalid child index for PatchedArray: {idx}"),
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn metadata(array: &Self::Array) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(PatchedMetadata {
            inner_len: array.inner.len() as u64,
            offset: array.offset as u32,
            n_patches: array.indices.len() as u32,
            n_lanes: array.n_lanes as u32,
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.serialize()))
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

    fn append_to_builder(
        array: &Self::Array,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let dtype = array.dtype();

        if !dtype.is_primitive() {
            // Default pathway: canonicalize and propagate.
            let canonical = array
                .clone()
                .into_array()
                .execute::<Canonical>(ctx)?
                .into_array();
            builder.extend_from_array(&canonical);
            return Ok(());
        }

        let ptype = dtype.as_ptype();

        let len = array.len();

        // Slice the inner by its offset before appending it.
        let sliced_inner = array.inner.slice(array.offset..array.offset + array.len)?;
        sliced_inner.append_to_builder(builder, ctx)?;

        let offset = array.offset;
        let lane_offsets: Buffer<u32> =
            Buffer::from_byte_buffer(array.lane_offsets.clone().unwrap_host());
        let indices = array.indices.clone().execute::<PrimitiveArray>(ctx)?;
        let values = array.values.clone().execute::<PrimitiveArray>(ctx)?;

        match_each_native_ptype!(ptype, |V| {
            let typed_builder = builder
                .as_any_mut()
                .downcast_mut::<PrimitiveBuilder<V>>()
                .vortex_expect("correctly typed builder");

            // Overwrite the last `len` elements of the builder. These would have been
            // populated by the inner.append_to_builder() call above.
            let output = typed_builder.values_mut();
            let trailer = output.len() - len;

            apply_patches_primitive::<V>(
                &mut output[trailer..],
                offset,
                len,
                array.n_chunks,
                array.n_lanes,
                &lane_offsets,
                indices.as_slice::<u16>(),
                values.as_slice::<V>(),
            );
        });

        Ok(())
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<PatchedArray> {
        let inner_len = usize::try_from(metadata.inner_len).map_err(|_| {
            vortex_err!(
                "PatchedMetadata inner_len overflows usize: {}",
                metadata.inner_len
            )
        })?;
        let offset = metadata.offset as usize;

        // n_chunks should correspond to the chunk in the `inner`.
        // After slicing when offset > 0, there may be additional chunks.
        let n_chunks = (len + offset).div_ceil(1024);
        let n_lanes = metadata.n_lanes as usize;

        let &[lane_offsets] = &buffers else {
            vortex_bail!("invalid buffer count for PatchedArray");
        };

        let inner = children.get(0, dtype, inner_len)?;
        let indices = children.get(1, PType::U16.into(), metadata.n_patches as usize)?;
        let values = children.get(2, dtype, metadata.n_patches as usize)?;

        Ok(PatchedArray {
            inner,
            n_chunks,
            n_lanes,
            offset,
            len,
            lane_offsets: lane_offsets.clone(),
            indices,
            values,
            stats_set: ArrayStats::default(),
        })
    }

    fn with_children(array: &mut Self::Array, mut children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 3,
            "PatchedArray must have exactly 3 children"
        );

        array.inner = children.remove(0);
        array.indices = children.remove(0);
        array.values = children.remove(0);

        Ok(())
    }

    fn execute(array: Arc<Array<Self>>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
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
        let indices = array.indices.clone().execute::<PrimitiveArray>(ctx)?;

        // TODO(aduffy): add support for non-primitive PatchedArray patches application (?)
        let values = array.values.clone().execute::<PrimitiveArray>(ctx)?;

        let patched_values = match_each_native_ptype!(values.ptype(), |V| {
            let offset = array.offset;
            let len = array.len;

            // Slice the buffer and validity from the offset.
            let buffer = buffer.slice_typed::<V>(offset..offset + len);
            let validity = validity.slice(offset..offset + len)?;

            let mut output = Buffer::<V>::from_byte_buffer(buffer.unwrap_host()).into_mut();

            apply_patches_primitive::<V>(
                &mut output,
                offset,
                len,
                array.n_chunks,
                array.n_lanes,
                &lane_offsets,
                indices.as_slice::<u16>(),
                values.as_slice::<V>(),
            );

            let output = output.freeze();

            PrimitiveArray::from_byte_buffer(output.into_byte_buffer(), ptype, validity)
        });

        Ok(ExecutionResult::done(patched_values.into_array()))
    }

    fn execute_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn reduce_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}

/// Apply patches on top of the existing value types.
#[allow(clippy::too_many_arguments)]
fn apply_patches_primitive<V: NativePType>(
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
            output[index - offset] = value;
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
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builders::builder_with_capacity;
    use crate::patches::Patches;
    use crate::validity::Validity;

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
    fn test_execute_sliced() {
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
            .slice(3..1024)
            .unwrap();

        let executed = array
            .execute::<Canonical>(&mut ctx)
            .unwrap()
            .into_primitive()
            .into_buffer::<u16>();

        let mut expected = buffer_mut![0u16; 1021];
        expected[0] = 1;

        assert_eq!(executed, expected.freeze());
    }

    #[test]
    fn test_append_to_builder_non_nullable() {
        let values = PrimitiveArray::new(buffer![0u16; 1024], Validity::NonNullable).into_array();
        let patches = Patches::new(
            1024,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![10u16, 20, 30].into_array(),
            None,
        )
        .unwrap();

        let session = VortexSession::empty();
        let mut ctx = ExecutionCtx::new(session);

        let array = PatchedArray::from_array_and_patches(values, &patches, &mut ctx)
            .unwrap()
            .into_array();

        let mut builder = builder_with_capacity(array.dtype(), array.len());
        array.append_to_builder(builder.as_mut(), &mut ctx).unwrap();

        let result = builder.finish();

        let mut expected = buffer_mut![0u16; 1024];
        expected[1] = 10;
        expected[2] = 20;
        expected[3] = 30;
        let expected = expected.into_array();

        assert_arrays_eq!(expected, result);
    }

    #[test]
    fn test_append_to_builder_sliced() {
        let values = PrimitiveArray::new(buffer![0u16; 1024], Validity::NonNullable).into_array();
        let patches = Patches::new(
            1024,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![10u16, 20, 30].into_array(),
            None,
        )
        .unwrap();

        let session = VortexSession::empty();
        let mut ctx = ExecutionCtx::new(session);

        let array = PatchedArray::from_array_and_patches(values, &patches, &mut ctx)
            .unwrap()
            .slice(3..1024)
            .unwrap();

        let mut builder = builder_with_capacity(array.dtype(), array.len());
        array.append_to_builder(builder.as_mut(), &mut ctx).unwrap();

        let result = builder.finish();

        let mut expected = buffer_mut![0u16; 1021];
        expected[0] = 30;
        let expected = expected.into_array();

        assert_arrays_eq!(expected, result);
    }

    #[test]
    fn test_append_to_builder_with_validity() {
        // Create inner array with nulls at indices 0 and 5.
        let validity = Validity::from_iter((0..10).map(|i| i != 0 && i != 5));
        let values = PrimitiveArray::new(buffer![0u16; 10], validity).into_array();

        // Apply patches at indices 1, 2, 3.
        let patches = Patches::new(
            10,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![10u16, 20, 30].into_array(),
            None,
        )
        .unwrap();

        let session = VortexSession::empty();
        let mut ctx = ExecutionCtx::new(session);

        let array = PatchedArray::from_array_and_patches(values, &patches, &mut ctx)
            .unwrap()
            .into_array();

        let mut builder = builder_with_capacity(array.dtype(), array.len());
        array.append_to_builder(builder.as_mut(), &mut ctx).unwrap();

        let result = builder.finish();

        // Expected: null at 0, patched 10/20/30 at 1/2/3, zero at 4, null at 5, zeros at 6-9.
        let expected = PrimitiveArray::from_option_iter([
            None,
            Some(10u16),
            Some(20),
            Some(30),
            Some(0),
            None,
            Some(0),
            Some(0),
            Some(0),
            Some(0),
        ])
        .into_array();

        assert_arrays_eq!(expected, result);
    }
}
