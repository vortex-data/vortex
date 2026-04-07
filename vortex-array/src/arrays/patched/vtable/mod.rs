// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message;

use crate::ArrayEq;
use crate::ArrayHash;
mod kernels;
mod operations;
mod slice;

use std::hash::Hash;
use std::hash::Hasher;

use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::Precision;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::array::ValidityChild;
use crate::array::ValidityVTableFromChild;
use crate::arrays::PrimitiveArray;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedData;
use crate::arrays::patched::array::SLOT_NAMES;
use crate::arrays::patched::compute::rules::PARENT_RULES;
use crate::arrays::patched::vtable::kernels::PARENT_KERNELS;
use crate::arrays::primitive::PrimitiveDataParts;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::PrimitiveBuilder;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::serde::ArrayChildren;
use crate::vtable;

vtable!(Patched, Patched, PatchedData);

#[derive(Clone, Debug)]
pub struct Patched;

impl ValidityChild<Patched> for Patched {
    fn validity_child(array: ArrayView<'_, Patched>) -> ArrayRef {
        array.base_array().clone()
    }
}

#[derive(Clone, prost::Message)]
pub struct PatchedMetadata {
    /// The total number of patches, and the length of the indices and values child arrays.
    #[prost(uint32, tag = "1")]
    pub(crate) n_patches: u32,

    /// The number of lanes used for patch indexing. Must be a power of two between 1 and 128.
    #[prost(uint32, tag = "2")]
    pub(crate) n_lanes: u32,

    /// An offset into the first chunk's patches that should be considered in-view.
    ///
    /// Always between 0 and 1023.
    #[prost(uint32, tag = "3")]
    pub(crate) offset: u32,
}

impl ArrayHash for PatchedData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _precision: Precision) {
        self.offset.hash(state);
        self.n_lanes.hash(state);
    }
}

impl ArrayEq for PatchedData {
    fn array_eq(&self, other: &Self, _precision: Precision) -> bool {
        self.offset == other.offset && self.n_lanes == other.n_lanes
    }
}

impl VTable for Patched {
    type ArrayData = PatchedData;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("vortex.patched")
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn validate(
        &self,
        data: &PatchedData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        data.validate(dtype, len, slots)
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("invalid buffer index for PatchedArray: {idx}");
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("invalid buffer index for PatchedArray: {idx}");
    }

    fn child(array: ArrayView<'_, Self>, idx: usize) -> ArrayRef {
        match idx {
            0 => array.base_array().clone(),
            1 => array.lane_offsets().clone(),
            2 => array.patch_indices().clone(),
            3 => array.patch_values().clone(),
            _ => vortex_panic!("invalid child index for PatchedArray: {idx}"),
        }
    }

    fn serialize(array: ArrayView<'_, Self>) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            PatchedMetadata {
                n_patches: u32::try_from(array.patch_indices().len())?,
                n_lanes: u32::try_from(array.n_lanes())?,
                offset: u32::try_from(array.offset())?,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = PatchedMetadata::decode(metadata)?;
        let n_patches = metadata.n_patches as usize;
        let n_lanes = metadata.n_lanes as usize;
        let offset = metadata.offset as usize;

        // n_chunks should correspond to the chunk in the `inner`.
        // After slicing when offset > 0, there may be additional chunks.
        let n_chunks = (len + offset).div_ceil(1024);

        let inner = children.get(0, dtype, len)?;
        let lane_offsets = children.get(1, PType::U32.into(), n_chunks * n_lanes + 1)?;
        let indices = children.get(2, PType::U16.into(), n_patches)?;
        let values = children.get(3, dtype, n_patches)?;

        let data = PatchedData { n_lanes, offset };
        Ok(
            ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(vec![
                Some(inner),
                Some(lane_offsets),
                Some(indices),
                Some(values),
            ]),
        )
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let dtype = array.array().dtype();

        if !dtype.is_primitive() {
            // Default pathway: canonicalize and propagate.
            let canonical = array
                .array()
                .clone()
                .execute::<Canonical>(ctx)?
                .into_array();
            builder.extend_from_array(&canonical);
            return Ok(());
        }

        let ptype = dtype.as_ptype();

        let len = array.len();

        array.base_array().append_to_builder(builder, ctx)?;

        let offset = array.offset();
        let lane_offsets = array
            .lane_offsets()
            .clone()
            .execute::<PrimitiveArray>(ctx)?;
        let indices = array
            .patch_indices()
            .clone()
            .execute::<PrimitiveArray>(ctx)?;
        let values = array
            .patch_values()
            .clone()
            .execute::<PrimitiveArray>(ctx)?;

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
                array.n_lanes(),
                lane_offsets.as_slice::<u32>(),
                indices.as_slice::<u16>(),
                values.as_slice::<V>(),
            );
        });

        Ok(())
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let inner = array
            .base_array()
            .clone()
            .execute::<Canonical>(ctx)?
            .into_primitive();

        let PrimitiveDataParts {
            buffer,
            ptype,
            validity,
        } = inner.into_data_parts();

        let lane_offsets = array
            .lane_offsets()
            .clone()
            .execute::<PrimitiveArray>(ctx)?;
        let indices = array
            .patch_indices()
            .clone()
            .execute::<PrimitiveArray>(ctx)?;

        // TODO(aduffy): add support for non-primitive PatchedArray patches application (?)
        let values = array
            .patch_values()
            .clone()
            .execute::<PrimitiveArray>(ctx)?;

        let patched_values = match_each_native_ptype!(values.ptype(), |V| {
            let offset = array.offset();
            let len = array.len();

            let mut output = Buffer::<V>::from_byte_buffer(buffer.unwrap_host()).into_mut();

            apply_patches_primitive::<V>(
                &mut output,
                offset,
                len,
                array.n_lanes(),
                lane_offsets.as_slice::<u32>(),
                indices.as_slice::<u16>(),
                values.as_slice::<V>(),
            );

            let output = output.freeze();

            PrimitiveArray::from_byte_buffer(output.into_byte_buffer(), ptype, validity)
        });

        Ok(ExecutionResult::done(patched_values.into_array()))
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
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
    n_lanes: usize,
    lane_offsets: &[u32],
    indices: &[u16],
    values: &[V],
) {
    let n_chunks = (offset + len).div_ceil(1024);
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
    use rstest::rstest;
    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_buffer::buffer_mut;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use crate::ArrayContext;
    use crate::Canonical;
    use crate::ExecutionCtx;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::arrays::Patched;
    use crate::arrays::PatchedArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::patched::array::PatchedArrayExt;
    use crate::assert_arrays_eq;
    use crate::builders::builder_with_capacity;
    use crate::patches::Patches;
    use crate::serde::SerializeOptions;
    use crate::serde::SerializedArray;
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

        let array = Patched::from_array_and_patches(values, &patches, &mut ctx)
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

        let array = Patched::from_array_and_patches(values, &patches, &mut ctx)
            .unwrap()
            .into_array()
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

        let array = Patched::from_array_and_patches(values, &patches, &mut ctx)
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

        let array = Patched::from_array_and_patches(values, &patches, &mut ctx)
            .unwrap()
            .into_array()
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

        let array = Patched::from_array_and_patches(values, &patches, &mut ctx)
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

    fn make_patched_array(
        inner: impl IntoIterator<Item = u16>,
        patch_indices: &[u32],
        patch_values: &[u16],
    ) -> VortexResult<PatchedArray> {
        let values: Vec<u16> = inner.into_iter().collect();
        let len = values.len();
        let array = PrimitiveArray::from_iter(values).into_array();

        let indices = PrimitiveArray::from_iter(patch_indices.iter().copied()).into_array();
        let patch_vals = PrimitiveArray::from_iter(patch_values.iter().copied()).into_array();

        let patches = Patches::new(len, 0, indices, patch_vals, None)?;

        let session = VortexSession::empty();
        let mut ctx = ExecutionCtx::new(session);

        Patched::from_array_and_patches(array, &patches, &mut ctx)
    }

    #[rstest]
    #[case::basic(
        make_patched_array(vec![0u16; 1024], &[1, 2, 3], &[10, 20, 30]).unwrap().into_array()
    )]
    #[case::multi_chunk(
        make_patched_array(vec![0u16; 4096], &[100, 1500, 2500, 3500], &[11, 22, 33, 44]).unwrap().into_array()
    )]
    #[case::sliced({
        let arr = make_patched_array(vec![0u16; 1024], &[1, 2, 3], &[10, 20, 30]).unwrap();
        arr.into_array().slice(2..1024).unwrap()
    })]
    fn test_serde_roundtrip(#[case] array: crate::ArrayRef) {
        let dtype = array.dtype().clone();
        let len = array.len();

        let ctx = ArrayContext::empty();
        let serialized = array.serialize(&ctx, &SerializeOptions::default()).unwrap();

        // Concat into a single buffer.
        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let concat = concat.freeze();

        let parts = SerializedArray::try_from(concat).unwrap();
        let decoded = parts
            .decode(
                &dtype,
                len,
                &ReadContext::new(ctx.to_ids()),
                &LEGACY_SESSION,
            )
            .unwrap();

        assert!(decoded.is::<Patched>());
        assert_eq!(
            array.display_values().to_string(),
            decoded.display_values().to_string()
        );
    }

    #[test]
    fn test_with_slots_basic() -> VortexResult<()> {
        let array = make_patched_array(vec![0u16; 1024], &[1, 2, 3], &[10, 20, 30])?;

        // Get original children via accessor methods
        let inner = array.base_array().clone();
        let lane_offsets = array.lane_offsets().clone();
        let indices = array.patch_indices().clone();
        let values = array.patch_values().clone();

        // Create new PatchedArray with same children using with_slots
        let array_ref = array.into_array();
        let new_array = array_ref.clone().with_slots(vec![
            Some(inner),
            Some(lane_offsets),
            Some(indices),
            Some(values),
        ])?;

        assert!(new_array.is::<Patched>());
        assert_eq!(array_ref.len(), new_array.len());
        assert_eq!(array_ref.dtype(), new_array.dtype());

        // Execute both and compare results
        let mut ctx = ExecutionCtx::new(VortexSession::empty());
        let original_executed = array_ref.execute::<Canonical>(&mut ctx)?.into_primitive();
        let new_executed = new_array.execute::<Canonical>(&mut ctx)?.into_primitive();

        assert_arrays_eq!(original_executed, new_executed);

        Ok(())
    }

    #[test]
    fn test_with_slots_modified_inner() -> VortexResult<()> {
        let array = make_patched_array(vec![0u16; 10], &[1, 2, 3], &[10, 20, 30])?;

        // Create a different inner array (all 5s instead of 0s)
        let new_inner = PrimitiveArray::from_iter(vec![5u16; 10]).into_array();
        let lane_offsets = array.lane_offsets().clone();
        let indices = array.patch_indices().clone();
        let values = array.patch_values().clone();

        let array_ref = array.into_array();
        let new_array = array_ref.with_slots(vec![
            Some(new_inner),
            Some(lane_offsets),
            Some(indices),
            Some(values),
        ])?;

        // Execute and verify the inner values changed (except at patch positions)
        let mut ctx = ExecutionCtx::new(VortexSession::empty());
        let executed = new_array.execute::<Canonical>(&mut ctx)?.into_primitive();

        // Expected: all 5s except indices 1, 2, 3 which are patched to 10, 20, 30
        let expected = PrimitiveArray::from_iter([5u16, 10, 20, 30, 5, 5, 5, 5, 5, 5]);
        assert_arrays_eq!(expected, executed);

        Ok(())
    }
}
