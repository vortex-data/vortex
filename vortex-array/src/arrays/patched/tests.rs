// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod conformance;
mod slots;

use rstest::rstest;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use super::FloatPatchStrategy;
use super::PatchStrategy;
use super::Patched;
use super::PatchedArrayExt;
use super::PatchedArraySlotsExt;
use crate::ArrayContext;
use crate::ArrayDeserialization;
use crate::ArrayPlugin;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::PrimitiveArray;
use crate::arrays::slice::SliceKernel;
use crate::assert_arrays_eq;
use crate::patches::Patches;
use crate::serde::SerializeOptions;
use crate::serde::SerializedArray;
use crate::session::ArraySessionExt;

#[rstest]
#[case(false)]
#[case(true)]
fn sliced_replacements_roundtrip(#[case] indexed: bool) -> VortexResult<()> {
    let session = VortexSession::empty();
    session.arrays().register(Patched);
    let mut ctx = session.create_execution_ctx();
    let len = 2050;
    let offsets = indexed.then(|| buffer![0u16, 2, 3].into_array());
    let patches = Patches::new(
        len,
        0,
        buffer![1u16, 1023, 1024, 2048].into_array(),
        PrimitiveArray::from_option_iter([Some(1i32), Some(9), None, Some(8)]).into_array(),
        offsets,
        &mut ctx,
    )?;
    let base = PrimitiveArray::from_option_iter((0..len).map(|i| (i != 1023).then_some(5i32)));
    let array = Patched::from_array_and_patches(base.into_array(), &patches, &mut ctx)?;
    assert_eq!(array.patch_chunk_offsets().len(), 3);
    let slice = <Patched as SliceKernel>::slice(array.as_view(), 1000..2049, &mut ctx)?.unwrap();
    let expected = PrimitiveArray::from_option_iter((1000..2049).map(|i| match i {
        1023 => Some(9i32),
        1024 => None,
        2048 => Some(8),
        _ => Some(5),
    }));
    let patches = slice.as_::<Patched>().patches();
    assert_eq!(patches.offset_within_chunk(), Some(1));

    let wire = ArrayContext::empty().with_allowed_ids(
        session.arrays().registry().read(|map| map.keys().copied().collect()),
    );
    let mut bytes = ByteBufferMut::empty();
    for buf in slice.serialize(&wire, &session, &SerializeOptions::default())? {
        bytes.extend_from_slice(&buf);
    }
    let decoded = SerializedArray::try_from(bytes.freeze())?.decode(
        slice.dtype(), slice.len(), &ReadContext::new(wire.to_ids()), &session,
    )?;
    assert!(decoded.is::<Patched>());
    assert_eq!(decoded.execute_scalar(23, &mut ctx)?, expected.execute_scalar(23, &mut ctx)?);
    assert!(decoded.execute_scalar(24, &mut ctx)?.is_null());
    assert_arrays_eq!(decoded, expected, &mut ctx);
    Ok(())
}

#[test]
fn strategy_matches_dtype_family() -> VortexResult<()> {
    let session = VortexSession::empty();
    let mut ctx = session.create_execution_ctx();
    let base = buffer![0f32; 3].into_array();
    let patches = Patches::new(3, 0, buffer![2u8].into_array(), buffer![1f32].into_array(), None, &mut ctx)?;
    let array = Patched::from_array_and_patches(base, &patches, &mut ctx)?;
    assert_eq!(array.strategy(), PatchStrategy::Float(FloatPatchStrategy::Replace));
    let integer_base = buffer![0i32; 3].into_array();
    assert!(Patched::with_strategy(integer_base, &patches, array.strategy(), &mut ctx).is_err());
    assert_arrays_eq!(array, buffer![0f32, 0.0, 1.0].into_array(), &mut ctx);
    Ok(())
}

#[test]
fn wire_requires_core_slots_and_accepts_trailing_slots() -> VortexResult<()> {
    let session = VortexSession::empty();
    let mut ctx = session.create_execution_ctx();
    let base = buffer![2u16; 4].into_array();
    let patches = Patches::new(
        4,
        0,
        buffer![1u8].into_array(),
        buffer![8u16].into_array(),
        None,
        &mut ctx,
    )?;
    let array = Patched::from_array_and_patches(base, &patches, &mut ctx)?.into_array();
    let wire = ArrayPlugin::serialize(&Patched, &array, &session)?.unwrap();
    assert!(wire.buffers.is_empty());
    assert_eq!(wire.children.len(), 4);
    let truncated = &wire.children[..3];
    let parts = ArrayDeserialization::new(
        wire.serialized_id,
        array.dtype(),
        array.len(),
        &wire.metadata,
        &[],
        &truncated,
    );
    assert!(ArrayPlugin::deserialize(&Patched, parts, &session).is_err());
    let mut children = wire.children;
    children.push(buffer![42u8].into_array());
    let parts = ArrayDeserialization::new(
        wire.serialized_id,
        array.dtype(),
        array.len(),
        &wire.metadata,
        &[],
        &children,
    );
    let decoded = ArrayPlugin::deserialize(&Patched, parts, &session)?;
    assert_arrays_eq!(array, decoded, &mut ctx);

    // A future integer operation may depend on that fifth child. An older reader must
    // reject its discriminator even when it recognizes the dtype and all core slots.
    let mut metadata = wire.metadata;
    metadata.extend_from_slice(&[0x10, 0x7f]); // field 2 (integer), operation 127
    let parts = ArrayDeserialization::new(
        wire.serialized_id,
        array.dtype(),
        array.len(),
        &metadata,
        &[],
        &children,
    );
    let error = ArrayPlugin::deserialize(&Patched, parts, &session).unwrap_err();
    assert!(error.to_string().contains("Unknown integer patch strategy"));
    Ok(())
}
