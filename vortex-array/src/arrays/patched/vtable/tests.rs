// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::LazyLock;

use prost::Message;
use rstest::rstest;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_buffer::buffer_mut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use super::PatchedMetadata;
use crate::Array;
use crate::ArrayContext;
use crate::ArrayDeserialization;
use crate::ArrayParts;
use crate::ArrayPlugin;
use crate::ArrayRef;
use crate::ArraySlots;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::Patched;
use crate::arrays::PatchedArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::arrays::patched::PatchedData;
use crate::arrays::patched::PatchedSlots;
use crate::arrays::patched::PatchedSlotsView;
use crate::assert_arrays_eq;
use crate::buffer::BufferHandle;
use crate::builders::builder_with_capacity_in;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::optimizer::ArrayOptimizer;
use crate::patches::Patches;
use crate::serde::ArrayChildren;
use crate::serde::SerializeOptions;
use crate::serde::SerializedArray;
use crate::session::ArraySessionExt;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

#[test]
fn test_execute() {
    let mut ctx = SESSION.create_execution_ctx();
    let values = buffer![0u16; 1024].into_array();
    let patches = Patches::new(
        1024,
        0,
        buffer![1u32, 2, 3].into_array(),
        buffer![1u16; 3].into_array(),
        None,
        &mut ctx,
    )
    .unwrap();

    let mut ctx = SESSION.create_execution_ctx();

    let array = Patched::from_array_and_patches(values, &patches, &mut ctx)
        .unwrap()
        .into_array();

    let mut expected = buffer_mut![0u16; 1024];
    expected[1] = 1;
    expected[2] = 1;
    expected[3] = 1;

    assert_arrays_eq!(array, expected.into_array(), &mut ctx);
}

#[test]
fn test_execute_sliced() {
    let mut ctx = SESSION.create_execution_ctx();
    let values = buffer![0u16; 1024].into_array();
    let patches = Patches::new(
        1024,
        0,
        buffer![1u32, 2, 3].into_array(),
        buffer![1u16; 3].into_array(),
        None,
        &mut ctx,
    )
    .unwrap();

    let mut ctx = SESSION.create_execution_ctx();

    let array = Patched::from_array_and_patches(values, &patches, &mut ctx)
        .unwrap()
        .into_array()
        .slice(3..1024)
        .unwrap();

    let mut expected = buffer_mut![0u16; 1021];
    expected[0] = 1;

    assert_arrays_eq!(array, expected.into_array(), &mut ctx);
}

#[test]
fn test_append_to_builder_non_nullable() {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::new(buffer![0u16; 1024], Validity::NonNullable).into_array();
    let patches = Patches::new(
        1024,
        0,
        buffer![1u32, 2, 3].into_array(),
        buffer![10u16, 20, 30].into_array(),
        None,
        &mut ctx,
    )
    .unwrap();

    let mut ctx = SESSION.create_execution_ctx();

    let array = Patched::from_array_and_patches(values, &patches, &mut ctx)
        .unwrap()
        .into_array();

    let mut builder = builder_with_capacity_in(
        array.dtype(),
        array.len(),
        vortex_buffer::BufferAllocatorRef::static_ref(),
    );
    array.append_to_builder(builder.as_mut(), &mut ctx).unwrap();

    let result = builder.finish();

    let mut expected = buffer_mut![0u16; 1024];
    expected[1] = 10;
    expected[2] = 20;
    expected[3] = 30;
    let expected = expected.into_array();

    assert_arrays_eq!(result, expected, &mut ctx);
}

#[test]
fn test_append_to_builder_sliced() {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::new(buffer![0u16; 1024], Validity::NonNullable).into_array();
    let patches = Patches::new(
        1024,
        0,
        buffer![1u32, 2, 3].into_array(),
        buffer![10u16, 20, 30].into_array(),
        None,
        &mut ctx,
    )
    .unwrap();

    let mut ctx = SESSION.create_execution_ctx();

    let array = Patched::from_array_and_patches(values, &patches, &mut ctx)
        .unwrap()
        .into_array()
        .slice(3..1024)
        .unwrap();

    let mut builder = builder_with_capacity_in(
        array.dtype(),
        array.len(),
        vortex_buffer::BufferAllocatorRef::static_ref(),
    );
    array.append_to_builder(builder.as_mut(), &mut ctx).unwrap();

    let result = builder.finish();

    let mut expected = buffer_mut![0u16; 1021];
    expected[0] = 30;
    let expected = expected.into_array();

    assert_arrays_eq!(result, expected, &mut ctx);
}

#[test]
fn test_append_to_builder_with_validity() {
    let mut ctx = SESSION.create_execution_ctx();
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
        &mut ctx,
    )
    .unwrap();

    let mut ctx = SESSION.create_execution_ctx();

    let array = Patched::from_array_and_patches(values, &patches, &mut ctx)
        .unwrap()
        .into_array();

    let mut builder = builder_with_capacity_in(
        array.dtype(),
        array.len(),
        vortex_buffer::BufferAllocatorRef::static_ref(),
    );
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

    assert_arrays_eq!(result, expected, &mut ctx);
}

fn make_patched_array(
    inner: impl IntoIterator<Item = u16>,
    patch_indices: &[u32],
    patch_values: &[u16],
) -> VortexResult<PatchedArray> {
    let mut ctx = SESSION.create_execution_ctx();
    let values: Vec<u16> = inner.into_iter().collect();
    let len = values.len();
    let array = PrimitiveArray::from_iter(values).into_array();

    let indices = PrimitiveArray::from_iter(patch_indices.iter().copied()).into_array();
    let patch_vals = PrimitiveArray::from_iter(patch_values.iter().copied()).into_array();

    let patches = Patches::new(len, 0, indices, patch_vals, None, &mut ctx)?;

    let mut ctx = SESSION.create_execution_ctx();

    Patched::from_array_and_patches(array, &patches, &mut ctx)
}

#[rstest]
#[case::basic(
    make_patched_array(vec![0u16; 1024], &[1, 2, 3], &[10, 20, 30]).unwrap().into_array(),
    (0..1024).map(|i| match i { 1 => 10u16, 2 => 20, 3 => 30, _ => 0 }).collect::<Vec<_>>()
)]
#[case::multi_chunk(
    make_patched_array(vec![0u16; 4096], &[100, 1500, 2500, 3500], &[11, 22, 33, 44]).unwrap().into_array(),
    (0..4096).map(|i| match i { 100 => 11u16, 1500 => 22, 2500 => 33, 3500 => 44, _ => 0 }).collect::<Vec<_>>()
)]
#[case::sliced({
    let arr = make_patched_array(vec![0u16; 1024], &[1, 2, 3], &[10, 20, 30]).unwrap();
    arr.into_array().slice(2..1024).unwrap().optimize_ctx(&SESSION).unwrap()
}, (2..1024).map(|i| match i { 2 => 20u16, 3 => 30, _ => 0 }).collect::<Vec<_>>())]
fn test_serde_roundtrip(#[case] array: ArrayRef, #[case] expected: Vec<u16>) {
    let dtype = array.dtype().clone();
    let len = array.len();

    let session = VortexSession::empty();
    session.arrays().register(Patched);

    let ctx = ArrayContext::empty().with_allowed_ids(
        session
            .arrays()
            .registry()
            .read(|map| map.keys().copied().collect()),
    );
    let serialized = array
        .serialize(&ctx, &session, &SerializeOptions::default())
        .unwrap();

    // Concat into a single buffer.
    let mut concat = ByteBufferMut::empty();
    for buf in serialized {
        concat.extend_from_slice(buf.as_ref());
    }
    let concat = concat.freeze();

    let parts = SerializedArray::try_from(concat).unwrap();
    let decoded = parts
        .decode(&dtype, len, &ReadContext::new(ctx.to_ids()), &session)
        .unwrap();

    assert!(decoded.is::<Patched>());
    let expected = PrimitiveArray::from_iter(expected);
    let mut exec_ctx = session.create_execution_ctx();
    assert_arrays_eq!(array, &expected, &mut exec_ctx);
    assert_arrays_eq!(decoded, expected, &mut exec_ctx);
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

    let mut ctx = SESSION.create_execution_ctx();

    let expected = PrimitiveArray::from_iter((0..1024).map(|i| match i {
        1 => 10u16,
        2 => 20,
        3 => 30,
        _ => 0,
    }));
    assert_arrays_eq!(new_array, expected, &mut ctx);

    Ok(())
}

#[test]
fn test_rebuild_modified_inner_from_parts() -> VortexResult<()> {
    let array = make_patched_array(vec![0u16; 10], &[1, 2, 3], &[10, 20, 30])?;

    // Create a different inner array (all 5s instead of 0s)
    let new_inner = PrimitiveArray::from_iter(vec![5u16; 10]).into_array();
    let slots = PatchedSlots {
        inner: new_inner,
        lane_offsets: array.lane_offsets().clone(),
        patch_indices: array.patch_indices().clone(),
        patch_values: array.patch_values().clone(),
    };

    let data = PatchedData {
        n_lanes: array.n_lanes(),
        offset: array.offset(),
    };
    let new_array = Array::try_from_parts(ArrayParts::new(
        Patched,
        array.dtype().clone(),
        array.len(),
        data,
        slots.into_slots(),
    ))?
    .into_array();

    // Execute and verify the inner values changed (except at patch positions)
    let mut ctx = SESSION.create_execution_ctx();

    // Expected: all 5s except indices 1, 2, 3 which are patched to 10, 20, 30
    let expected = PrimitiveArray::from_iter([5u16, 10, 20, 30, 5, 5, 5, 5, 5, 5]);
    assert_arrays_eq!(new_array, expected, &mut ctx);

    Ok(())
}

fn serde_roundtrip(array: &ArrayRef) -> VortexResult<ArrayRef> {
    let session = VortexSession::empty();
    session.arrays().register(Patched);
    let ctx = ArrayContext::empty().with_allowed_ids(
        session
            .arrays()
            .registry()
            .read(|map| map.keys().copied().collect()),
    );
    let mut bytes = ByteBufferMut::empty();
    for buffer in array.serialize(&ctx, &session, &SerializeOptions::default())? {
        bytes.extend_from_slice(&buffer);
    }
    SerializedArray::try_from(bytes.freeze())?.decode(
        array.dtype(),
        array.len(),
        &ReadContext::new(ctx.to_ids()),
        &session,
    )
}

/// A nullable array whose patches all land on valid positions.
fn nullable_patched_array(len: usize) -> VortexResult<PatchedArray> {
    let mut ctx = SESSION.create_execution_ctx();
    let inner =
        PrimitiveArray::from_option_iter((0..len).map(|index| (index % 5 != 0).then_some(0u16)))
            .into_array();
    let patches = Patches::new(
        len,
        0,
        buffer![1u32, 1031, 1499].into_array(),
        buffer![10u16, 20, 30].into_array(),
        None,
        &mut ctx,
    )?;
    Patched::from_array_and_patches(inner, &patches, &mut ctx)
}

fn nullable_patched_expected(range: Range<usize>) -> PrimitiveArray {
    PrimitiveArray::from_option_iter(range.map(|index| {
        (index % 5 != 0).then_some(match index {
            1 => 10u16,
            1031 => 20,
            1499 => 30,
            _ => 0,
        })
    }))
}

/// A nullable array sliced into its second chunk round-trips through serde with its offset.
#[test]
fn test_serde_roundtrip_nullable_sliced() -> VortexResult<()> {
    let array = nullable_patched_array(1500)?
        .into_array()
        .slice(1030..1500)?
        .optimize_ctx(&SESSION)?;
    let patched = array
        .as_opt::<Patched>()
        .ok_or_else(|| vortex_err!("expected the slice to stay Patched"))?;
    assert_eq!(patched.offset(), 6);

    let restored = serde_roundtrip(&array)?;
    let restored_patched = restored
        .as_opt::<Patched>()
        .ok_or_else(|| vortex_err!("expected deserialized Patched"))?;
    assert_eq!(restored_patched.offset(), 6);
    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(restored, nullable_patched_expected(1030..1500), &mut ctx);
    Ok(())
}

fn deserialize_patched(
    dtype: &DType,
    len: usize,
    metadata: &PatchedMetadata,
    buffers: &[BufferHandle],
    children: &dyn ArrayChildren,
) -> VortexResult<ArrayRef> {
    ArrayPlugin::deserialize(
        &Patched,
        ArrayDeserialization::new(
            ArrayPlugin::id(&Patched),
            dtype,
            len,
            &metadata.encode_to_vec(),
            buffers,
            children,
        ),
        &SESSION,
    )
}

fn patched_children(array: &PatchedArray) -> [ArrayRef; 4] {
    [
        array.inner().clone(),
        array.lane_offsets().clone(),
        array.patch_indices().clone(),
        array.patch_values().clone(),
    ]
}

#[test]
fn test_deserialize_rejects_malformed_parts() -> VortexResult<()> {
    let array = nullable_patched_array(1500)?;
    let dtype = array.dtype().clone();
    let len = array.len();
    let metadata = PatchedMetadata {
        n_patches: 3,
        n_lanes: u32::try_from(array.n_lanes())?,
        offset: 0,
    };
    let children = patched_children(&array);
    let deserialize = |dtype: &DType, len: usize, metadata: &PatchedMetadata| {
        deserialize_patched(dtype, len, metadata, &[], &children)
    };
    let mut ctx = SESSION.create_execution_ctx();

    let restored = deserialize(&dtype, len, &metadata)?;
    assert_arrays_eq!(restored, nullable_patched_expected(0..len), &mut ctx);

    // Readers ignore children that follow the ones the contract declares.
    let [inner, lane_offsets, patch_indices, patch_values] = children.clone();
    let with_trailing = [
        inner.clone(),
        lane_offsets.clone(),
        patch_indices.clone(),
        patch_values,
        inner.clone(),
    ];
    let restored = deserialize_patched(&dtype, len, &metadata, &[], &with_trailing)?;
    assert_arrays_eq!(restored, nullable_patched_expected(0..len), &mut ctx);

    // A missing child.
    let missing = [inner, lane_offsets, patch_indices];
    assert!(deserialize_patched(&dtype, len, &metadata, &[], &missing).is_err());

    // Patched arrays have no buffers.
    let buffers = [BufferHandle::new_host(buffer![0u8])];
    assert!(deserialize_patched(&dtype, len, &metadata, &buffers, &children).is_err());

    // A non-primitive dtype.
    assert!(deserialize(&DType::Utf8(Nullability::Nullable), len, &metadata).is_err());

    // The lane count must be a power of two between 1 and 128.
    for n_lanes in [0, 3, 256] {
        let bad_lanes = PatchedMetadata {
            n_lanes,
            ..metadata.clone()
        };
        assert!(deserialize(&dtype, len, &bad_lanes).is_err());
    }

    // The offset must lie in the first chunk.
    let offset_past_chunk = PatchedMetadata {
        offset: 1024,
        ..metadata.clone()
    };
    assert!(deserialize(&dtype, len, &offset_past_chunk).is_err());

    // The offset plus the length must not overflow.
    let offset = PatchedMetadata {
        offset: 1,
        ..metadata.clone()
    };
    assert!(deserialize(&dtype, usize::MAX, &offset).is_err());
    Ok(())
}

/// Children that report a dtype or length mismatch as an error, as serialized children do,
/// rather than panicking as the in-memory [`ArrayChildren`] impl does.
struct CheckedChildren<'a>(&'a [ArrayRef]);

impl ArrayChildren for CheckedChildren<'_> {
    fn get(&self, index: usize, dtype: &DType, len: usize) -> VortexResult<ArrayRef> {
        let child = self
            .0
            .get(index)
            .ok_or_else(|| vortex_err!("missing child {index}"))?;
        vortex_ensure_eq!(child.dtype(), dtype);
        vortex_ensure_eq!(child.len(), len);
        Ok(child.clone())
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

/// The reader asks for each child with the dtype and length the contract derives for it.
#[test]
fn test_deserialize_rejects_mismatched_children() -> VortexResult<()> {
    let array = nullable_patched_array(1500)?;
    let dtype = array.dtype().clone();
    let len = array.len();
    let metadata = PatchedMetadata {
        n_patches: 3,
        n_lanes: u32::try_from(array.n_lanes())?,
        offset: 0,
    };
    let children = patched_children(&array);
    let deserialize = |metadata: &PatchedMetadata, children: &[ArrayRef]| {
        deserialize_patched(&dtype, len, metadata, &[], &CheckedChildren(children))
    };

    assert!(deserialize(&metadata, &children).is_ok());

    // `n_patches` must match the patch indices and values.
    for n_patches in [2, 4] {
        let wrong_count = PatchedMetadata {
            n_patches,
            ..metadata.clone()
        };
        assert!(deserialize(&wrong_count, &children).is_err());
    }

    // The lane count fixes the length of the lane offsets.
    let fewer_lanes = PatchedMetadata {
        n_lanes: metadata.n_lanes / 2,
        ..metadata.clone()
    };
    assert!(deserialize(&fewer_lanes, &children).is_err());

    // Patch indices must be `u16`.
    let mut wide_indices = children.clone();
    wide_indices[PatchedSlots::PATCH_INDICES] = wide_indices[PatchedSlots::PATCH_INDICES]
        .cast(DType::Primitive(PType::U32, Nullability::NonNullable))?;
    assert!(deserialize(&metadata, &wide_indices).is_err());

    // Patch values must be non-nullable, since patching ignores their validity.
    let mut nullable_values = children;
    nullable_values[PatchedSlots::PATCH_VALUES] =
        nullable_values[PatchedSlots::PATCH_VALUES].cast(dtype.clone())?;
    assert!(deserialize(&metadata, &nullable_values).is_err());
    Ok(())
}

#[test]
fn test_try_from_parts_rejects_malformed_slots() -> VortexResult<()> {
    let array = nullable_patched_array(1500)?;
    let data = PatchedData {
        n_lanes: array.n_lanes(),
        offset: array.offset(),
    };
    let try_from_slots = |data: PatchedData, slots: [ArrayRef; 4]| {
        let [inner, lane_offsets, patch_indices, patch_values] = slots;
        let slots = PatchedSlots {
            inner,
            lane_offsets,
            patch_indices,
            patch_values,
        };
        Array::try_from_parts(ArrayParts::new(
            Patched,
            array.dtype().clone(),
            array.len(),
            data,
            slots.into_slots(),
        ))
    };
    let children = patched_children(&array);

    assert!(try_from_slots(data.clone(), children.clone()).is_ok());

    let zero_lanes = PatchedData {
        n_lanes: 0,
        ..data.clone()
    };
    assert!(try_from_slots(zero_lanes, children.clone()).is_err());

    let mut short_lane_offsets = children.clone();
    let lane_offsets_len = short_lane_offsets[PatchedSlots::LANE_OFFSETS].len();
    short_lane_offsets[PatchedSlots::LANE_OFFSETS] =
        short_lane_offsets[PatchedSlots::LANE_OFFSETS].slice(0..lane_offsets_len - 1)?;
    assert!(try_from_slots(data.clone(), short_lane_offsets).is_err());

    let mut nullable_values = children;
    nullable_values[PatchedSlots::PATCH_VALUES] =
        nullable_values[PatchedSlots::PATCH_VALUES].cast(array.dtype().clone())?;
    assert!(try_from_slots(data, nullable_values).is_err());
    Ok(())
}
