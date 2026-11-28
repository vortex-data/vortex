// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! BitPacking scheme selection and decline cases (empty input, full-width values, patches).

#![cfg(test)]

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ArrayVTable;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Patched;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::session::ArraySessionExt;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::schemes::integer::BitPackingScheme;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPacked;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    session.arrays().register(Patched);
    vortex_fastlanes::initialize(&session);
    session
});

/// Values that need 7 bits in every block.
fn uniform() -> ArrayRef {
    PrimitiveArray::from_iter((0..8192u32).map(|i| i % 128)).into_array()
}

/// Compresses `array`, round trips it through serialization, and returns the serialized IDs.
fn compress_roundtrip(
    builder: BtrBlocksCompressorBuilder,
    array: &ArrayRef,
) -> VortexResult<Vec<ArrayId>> {
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = builder.build().compress(array, &mut ctx)?;
    assert_arrays_eq!(array, compressed, &mut ctx);

    let array_ctx = ArrayContext::empty();
    let mut bytes = ByteBufferMut::empty();
    for buffer in compressed.serialize(&array_ctx, &SESSION, &SerializeOptions::default())? {
        bytes.extend_from_slice(&buffer);
    }
    let read = SerializedArray::try_from(bytes.freeze())?.decode(
        array.dtype(),
        array.len(),
        &ReadContext::new(array_ctx.to_ids()),
        &SESSION,
    )?;
    assert_arrays_eq!(array, read, &mut ctx);
    Ok(array_ctx.to_ids())
}

fn bitpacking_only() -> BtrBlocksCompressorBuilder {
    BtrBlocksCompressorBuilder::empty().with_new_scheme(&BitPackingScheme)
}

#[rstest]
#[case::uniform(uniform(), true)]
#[case::uniform_with_patches(
    PrimitiveArray::from_iter(
        (0..8192u32).map(|i| if i % 1024 == 0 { 1 << 20 } else { i % 128 }),
    )
    .into_array(),
    true
)]
#[case::native_u8(
    PrimitiveArray::from_iter((128u8..=u8::MAX).cycle().take(8192)).into_array(),
    false
)]
#[case::native_u32(
    PrimitiveArray::from_iter((0..8192u32).map(|i| (1 << 31) + i)).into_array(),
    false
)]
#[case::empty(PrimitiveArray::from_iter(Vec::<u32>::new()).into_array(), false)]
fn selects_bitpacked_or_declines(
    #[case] array: ArrayRef,
    #[case] expect_bitpacked: bool,
) -> VortexResult<()> {
    let ids = compress_roundtrip(bitpacking_only(), &array)?;
    assert_eq!(ids.contains(&BitPacked.id()), expect_bitpacked);
    Ok(())
}
