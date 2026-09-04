// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernels;
mod operations;
mod slice;

use std::hash::Hash;
use std::hash::Hasher;

use prost::Message;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::Canonical;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::array::ValidityChild;
use crate::array::ValidityVTableFromChild;
use crate::array::with_empty_buffers;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::arrays::patched::PatchedData;
use crate::arrays::patched::PatchedSlots;
use crate::arrays::patched::PatchedSlotsView;
use crate::arrays::patched::PatchedView;
use crate::arrays::patched::compute::rules::PARENT_RULES;
use crate::arrays::patched::layout::n_chunks;
use crate::arrays::primitive::PrimitiveDataParts;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::PrimitiveBuilder;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::require_child;
use crate::serde::ArrayChildren;

/// A [`Patched`]-encoded Vortex array.
pub type PatchedArray = Array<Patched>;

pub(crate) fn initialize(session: &VortexSession) {
    kernels::initialize(session);
}

#[derive(Clone, Debug)]
pub struct Patched;

impl ValidityChild<Patched> for Patched {
    fn validity_child(array: ArrayView<'_, Patched>) -> ArrayRef {
        array.inner().clone()
    }
}

/// Metadata of the chunk-local layout, written under the `vortex.patched_v2` wire ID.
#[derive(Clone, prost::Message)]
pub struct PatchedMetadata {
    /// The number of patches, and the length of the `patch_indices` and `patch_values` children.
    #[prost(uint32, tag = "1")]
    pub(crate) n_patches: u32,

    /// Grid position of logical row zero. Always below 1024.
    #[prost(uint32, tag = "2")]
    pub(crate) offset: u32,
}

impl ArrayHash for PatchedData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.offset.hash(state);
    }
}

impl ArrayEq for PatchedData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.offset == other.offset
    }
}

impl VTable for Patched {
    type TypedArrayData = PatchedData;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.patched");
        *ID
    }

    fn validate(
        &self,
        data: &PatchedData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        data.validate(dtype, len, &PatchedSlotsView::from_slots(slots))
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("invalid buffer index for PatchedArray: {idx}");
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("invalid buffer index for PatchedArray: {idx}");
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            PatchedMetadata {
                n_patches: u32::try_from(array.patch_indices().len())?,
                offset: u32::try_from(array.offset())?,
            }
            .encode_to_vec(),
        ))
    }

    /// Deserialize the chunk-local layout. The retired lane layout under `vortex.patched` is
    /// handled by [`PatchedPlugin`](crate::arrays::patched::PatchedPlugin).
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
        let offset = metadata.offset as usize;

        let inner = children.get(0, dtype, len)?;
        let patch_indices = children.get(1, &PType::U16.into(), n_patches)?;
        let patch_values = children.get(2, dtype, n_patches)?;
        let chunk_offsets = children.get(3, &PType::U32.into(), n_chunks(offset, len) + 1)?;

        let slots = PatchedSlots {
            inner,
            patch_indices,
            patch_values,
            chunk_offsets,
        }
        .into_slots();
        Ok(
            ArrayParts::new(self.clone(), dtype.clone(), len, PatchedData { offset })
                .with_slots(slots),
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
            return canonical.append_to_builder(builder, ctx);
        }

        let ptype = dtype.as_ptype();
        let len = array.len();

        array.inner().append_to_builder(builder, ctx)?;

        let indices = array
            .patch_indices()
            .clone()
            .execute::<PrimitiveArray>(ctx)?;
        let values = array
            .patch_values()
            .clone()
            .execute::<PrimitiveArray>(ctx)?;
        let chunk_offsets = array
            .chunk_offsets()
            .clone()
            .execute::<PrimitiveArray>(ctx)?;
        let view = PatchedView::new(
            array.offset(),
            len,
            indices.as_slice::<u16>(),
            chunk_offsets.as_slice::<u32>(),
        );

        match_each_native_ptype!(ptype, |V| {
            let typed_builder = builder
                .as_any_mut()
                .downcast_mut::<PrimitiveBuilder<V>>()
                .vortex_expect("correctly typed builder");

            // Overwrite the last `len` elements of the builder. These would have been
            // populated by the inner.append_to_builder() call above.
            let output = typed_builder.values_mut();
            let trailer = output.len() - len;
            apply_patches_primitive::<V>(&mut output[trailer..], view, values.as_slice::<V>());
        });

        Ok(())
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        PatchedSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let array = require_child!(array, array.inner(), PatchedSlots::INNER => Primitive);
        let array =
            require_child!(array, array.patch_indices(), PatchedSlots::PATCH_INDICES => Primitive);
        let array =
            require_child!(array, array.patch_values(), PatchedSlots::PATCH_VALUES => Primitive);
        let array =
            require_child!(array, array.chunk_offsets(), PatchedSlots::CHUNK_OFFSETS => Primitive);

        let len = array.len();
        let offset = array.offset;
        let slots = match array.try_into_parts() {
            Ok(parts) => PatchedSlots::from_slots(parts.slots),
            Err(array) => PatchedSlotsView::from_slots(array.slots()).to_owned(),
        };

        // TODO(joe): use iterative execution
        let PrimitiveDataParts {
            buffer,
            ptype,
            validity,
        } = slots.inner.downcast::<Primitive>().into_data_parts();

        let values = slots.patch_values.downcast::<Primitive>();
        let indices = slots.patch_indices.downcast::<Primitive>();
        let chunk_offsets = slots.chunk_offsets.downcast::<Primitive>();
        let view = PatchedView::new(
            offset,
            len,
            indices.as_slice::<u16>(),
            chunk_offsets.as_slice::<u32>(),
        );

        let patched_values = match_each_native_ptype!(ptype, |V| {
            let mut output = Buffer::<V>::from_byte_buffer(buffer.unwrap_host()).into_mut();
            apply_patches_primitive::<V>(&mut output, view, values.as_slice::<V>());
            PrimitiveArray::from_byte_buffer(output.freeze().into_byte_buffer(), ptype, validity)
        });

        Ok(ExecutionResult::done(patched_values.into_array()))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}

/// Overwrite the patched rows of `output` with their patch values.
fn apply_patches_primitive<V: NativePType>(output: &mut [V], view: PatchedView<'_>, values: &[V]) {
    view.for_each(|row, ordinal| output[row] = values[ordinal]);
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use rstest::rstest;
    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_buffer::buffer_mut;
    use vortex_error::VortexResult;
    use vortex_session::registry::ReadContext;

    use crate::Array;
    use crate::ArrayContext;
    use crate::ArrayParts;
    use crate::ArraySlots;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::Patched;
    use crate::arrays::PatchedArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::patched::PatchedArrayExt;
    use crate::arrays::patched::PatchedArraySlotsExt;
    use crate::arrays::patched::PatchedData;
    use crate::arrays::patched::PatchedPlugin;
    use crate::arrays::patched::PatchedSlots;
    use crate::arrays::patched::PatchedSlotsView;
    use crate::assert_arrays_eq;
    use crate::builders::builder_with_capacity;
    use crate::patches::Patches;
    use crate::serde::SerializeOptions;
    use crate::serde::SerializedArray;
    use crate::session::ArraySessionExt;
    use crate::validity::Validity;

    #[test]
    fn test_execute() -> VortexResult<()> {
        let values = buffer![0u16; 1024].into_array();
        let patches = Patches::new(
            1024,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![1u16; 3].into_array(),
            None,
        )?;

        let session = array_session();
        let mut ctx = session.create_execution_ctx();

        let array = Patched::from_array_and_patches(values, &patches, &mut ctx)?.into_array();

        let executed = array
            .execute::<Canonical>(&mut ctx)?
            .into_primitive()
            .into_buffer::<u16>();

        let mut expected = buffer_mut![0u16; 1024];
        expected[1] = 1;
        expected[2] = 1;
        expected[3] = 1;

        assert_eq!(executed, expected.freeze());
        Ok(())
    }

    #[test]
    fn test_execute_sliced() -> VortexResult<()> {
        let values = buffer![0u16; 1024].into_array();
        let patches = Patches::new(
            1024,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![1u16; 3].into_array(),
            None,
        )?;

        let session = array_session();
        let mut ctx = session.create_execution_ctx();

        let array = Patched::from_array_and_patches(values, &patches, &mut ctx)?
            .into_array()
            .slice(3..1024)?;

        let executed = array
            .execute::<Canonical>(&mut ctx)?
            .into_primitive()
            .into_buffer::<u16>();

        let mut expected = buffer_mut![0u16; 1021];
        expected[0] = 1;

        assert_eq!(executed, expected.freeze());
        Ok(())
    }

    #[test]
    fn test_append_to_builder_non_nullable() -> VortexResult<()> {
        let values = PrimitiveArray::new(buffer![0u16; 1024], Validity::NonNullable).into_array();
        let patches = Patches::new(
            1024,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![10u16, 20, 30].into_array(),
            None,
        )?;

        let session = array_session();
        let mut ctx = session.create_execution_ctx();

        let array = Patched::from_array_and_patches(values, &patches, &mut ctx)?.into_array();

        let mut builder = builder_with_capacity(array.dtype(), array.len());
        array.append_to_builder(builder.as_mut(), &mut ctx)?;

        let result = builder.finish();

        let mut expected = buffer_mut![0u16; 1024];
        expected[1] = 10;
        expected[2] = 20;
        expected[3] = 30;
        let expected = expected.into_array();

        assert_arrays_eq!(expected, result, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_append_to_builder_sliced() -> VortexResult<()> {
        let values = PrimitiveArray::new(buffer![0u16; 1024], Validity::NonNullable).into_array();
        let patches = Patches::new(
            1024,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![10u16, 20, 30].into_array(),
            None,
        )?;

        let session = array_session();
        let mut ctx = session.create_execution_ctx();

        let array = Patched::from_array_and_patches(values, &patches, &mut ctx)?
            .into_array()
            .slice(3..1024)?;

        let mut builder = builder_with_capacity(array.dtype(), array.len());
        array.append_to_builder(builder.as_mut(), &mut ctx)?;

        let result = builder.finish();

        let mut expected = buffer_mut![0u16; 1021];
        expected[0] = 30;
        let expected = expected.into_array();

        assert_arrays_eq!(expected, result, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_append_to_builder_with_validity() -> VortexResult<()> {
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
        )?;

        let session = array_session();
        let mut ctx = session.create_execution_ctx();

        let array = Patched::from_array_and_patches(values, &patches, &mut ctx)?.into_array();

        let mut builder = builder_with_capacity(array.dtype(), array.len());
        array.append_to_builder(builder.as_mut(), &mut ctx)?;

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

        assert_arrays_eq!(expected, result, &mut ctx);
        Ok(())
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

        let session = array_session();
        let mut ctx = session.create_execution_ctx();

        Patched::from_array_and_patches(array, &patches, &mut ctx)
    }

    #[rstest]
    #[case::basic(1024, &[1, 2, 3], &[10, 20, 30], None)]
    #[case::multi_chunk(4096, &[100, 1500, 2500, 3500], &[11, 22, 33, 44], None)]
    #[case::sliced(1024, &[1, 2, 3], &[10, 20, 30], Some(2..1024))]
    // Rows 1030 and 3000 fall outside the slice but stay in the shared patch children.
    #[case::sliced_with_dead_patches(4096, &[1030, 1031, 2200, 3000], &[1, 2, 3, 4], Some(1031..3000))]
    fn test_serde_roundtrip(
        #[case] len: usize,
        #[case] patch_indices: &[u32],
        #[case] patch_values: &[u16],
        #[case] slice: Option<Range<usize>>,
    ) -> VortexResult<()> {
        let array = make_patched_array(vec![0u16; len], patch_indices, patch_values)?.into_array();
        let array = match slice {
            Some(range) => array.slice(range)?,
            None => array,
        };
        let dtype = array.dtype().clone();
        let len = array.len();

        let session = array_session();
        session.arrays().register(PatchedPlugin);

        let ctx = ArrayContext::empty().with_allowed_ids(
            session
                .arrays()
                .registry()
                .read(|map| map.keys().copied().collect()),
        );
        let serialized = array.serialize(&ctx, &session, &SerializeOptions::default())?;

        // Concat into a single buffer.
        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let concat = concat.freeze();

        let parts = SerializedArray::try_from(concat)?;
        let decoded = parts.decode(&dtype, len, &ReadContext::new(ctx.to_ids()), &session)?;

        assert!(decoded.is::<Patched>());
        assert_eq!(
            array.display_values().to_string(),
            decoded.display_values().to_string()
        );
        Ok(())
    }

    #[test]
    fn test_with_slots_basic() -> VortexResult<()> {
        let array = make_patched_array(vec![0u16; 1024], &[1, 2, 3], &[10, 20, 30])?;

        // Get original children via accessor methods
        let slots = PatchedSlots::from_slots(
            array
                .as_array()
                .slots()
                .iter()
                .cloned()
                .collect::<ArraySlots>(),
        );
        let view = PatchedSlotsView::from_slots(array.as_array().slots());
        assert_eq!(view.inner.len(), array.inner().len());

        // Create new PatchedArray with same children using with_slots
        let array_ref = array.into_array();
        // SAFETY: the replacement slots are the original children, preserving logical values and
        // parent statistics.
        let new_array = unsafe { array_ref.clone().with_slots(slots.into_slots()) }?;

        assert!(new_array.is::<Patched>());
        assert_eq!(array_ref.len(), new_array.len());
        assert_eq!(array_ref.dtype(), new_array.dtype());

        // Execute both and compare results
        let mut ctx = array_session().create_execution_ctx();
        let original_executed = array_ref.execute::<Canonical>(&mut ctx)?.into_primitive();
        let new_executed = new_array.execute::<Canonical>(&mut ctx)?.into_primitive();

        assert_arrays_eq!(original_executed, new_executed, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_rebuild_modified_inner_from_parts() -> VortexResult<()> {
        let array = make_patched_array(vec![0u16; 10], &[1, 2, 3], &[10, 20, 30])?;

        // Create a different inner array (all 5s instead of 0s)
        let new_inner = PrimitiveArray::from_iter(vec![5u16; 10]).into_array();
        let slots = PatchedSlots {
            inner: new_inner,
            patch_indices: array.patch_indices().clone(),
            patch_values: array.patch_values().clone(),
            chunk_offsets: array.chunk_offsets().clone(),
        };

        let data = PatchedData {
            offset: array.offset(),
        };
        let new_array = Array::try_from_parts(
            ArrayParts::new(Patched, array.dtype().clone(), array.len(), data)
                .with_slots(slots.into_slots()),
        )?
        .into_array();

        // Execute and verify the inner values changed (except at patch positions)
        let mut ctx = array_session().create_execution_ctx();
        let executed = new_array.execute::<Canonical>(&mut ctx)?.into_primitive();

        // Expected: all 5s except indices 1, 2, 3 which are patched to 10, 20, 30
        let expected = PrimitiveArray::from_iter([5u16, 10, 20, 30, 5, 5, 5, 5, 5, 5]);
        assert_arrays_eq!(expected, executed, &mut ctx);

        Ok(())
    }
}
