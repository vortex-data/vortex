// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Preview gating and codec cascades retain Narrow's logical dtype on the wire.
//!
//! Core and CUDA presets exclude Narrow wrappers. Serialization round trips also cover signed
//! internal buffers whose dtypes are selected by the compressor.

#![cfg(test)]

use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::Narrow;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::narrow::NarrowArraySlotsExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
#[cfg(feature = "pco")]
use vortex_array::session::ArraySessionExt;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::CompressionSession;
use vortex_buffer::ByteBufferMut;
use vortex_edition::ComponentKind;
use vortex_edition::EDITION_DECLARATIONS;
use vortex_edition::EDITION_FAMILIES;
use vortex_edition::EditionSessionExt;
use vortex_edition::declarations::core::CORE_2026_08_3;
use vortex_edition::declarations::preview::PREVIEW_2026_10_0;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

fn session(preview: bool) -> VortexResult<VortexSession> {
    let session = vortex_array::array_session().with::<CompressionSession>();
    vortex_fastlanes::initialize(&session);
    for family in EDITION_FAMILIES {
        session.editions().declare_family(family)?;
    }
    for declaration in EDITION_DECLARATIONS {
        session.register_edition(declaration)?;
    }
    session.enable_edition(CORE_2026_08_3)?;
    if preview {
        session.enable_edition(PREVIEW_2026_10_0)?;
    }

    Ok(session)
}

#[rstest]
#[case::core(false)]
#[case::preview(true)]
fn test_compression_roundtrip(#[case] preview: bool) -> VortexResult<()> {
    let session = session(preview)?;
    let mut ctx = session.create_execution_ctx();
    let input = PrimitiveArray::from_iter((0..8192u64).map(|i| i % 128)).into_array();
    let compressed = BtrBlocksCompressorBuilder::from_session(&session)
        .build()
        .compress(&input, &mut ctx)?;

    assert_eq!(compressed.is::<Narrow>(), preview);
    if preview {
        assert_eq!(
            compressed.as_::<Narrow>().values().dtype().as_ptype(),
            PType::U8
        );
    }
    assert_eq!(compressed.dtype(), input.dtype());
    assert!(compressed.nbytes() < input.nbytes());
    let array_ctx = ArrayContext::empty();
    let mut bytes = ByteBufferMut::empty();
    for buffer in compressed.serialize(&array_ctx, &session, &SerializeOptions::default())? {
        bytes.extend_from_slice(&buffer);
    }
    let ids = array_ctx.to_ids();
    let allowed = session.enabled_component_ids(ComponentKind::Array);
    assert!(ids.iter().all(|id| allowed.contains(id)));
    assert_eq!(ids.contains(&Narrow.id()), preview);
    let decoded = SerializedArray::try_from(bytes.freeze())?.decode(
        input.dtype(),
        input.len(),
        &ReadContext::new(ids),
        &session,
    )?;
    assert_arrays_eq!(decoded, input, &mut ctx);

    Ok(())
}

#[test]
fn test_cuda_preset_excludes_narrow() -> VortexResult<()> {
    let session = session(true)?;
    let mut ctx = session.create_execution_ctx();
    let input = PrimitiveArray::from_iter((0..8192u64).map(|i| i % 128)).into_array();
    let result = BtrBlocksCompressorBuilder::from_session(&session)
        .only_cuda_compatible()
        .build()
        .compress(&input, &mut ctx)?;
    assert!(!result.is::<Narrow>());
    assert_arrays_eq!(result, input, &mut ctx);

    Ok(())
}

#[test]
fn test_internal_signed_buffers_roundtrip() -> VortexResult<()> {
    let session = session(false)?;
    vortex_fsst::initialize(&session);
    #[cfg(feature = "pco")]
    session.arrays().register(vortex_pco::Pco);
    let mut ctx = session.create_execution_ctx();
    let elements = PrimitiveArray::from_iter((0..4096i32).map(|i| i % 7)).into_array();
    let offsets = PrimitiveArray::from_iter((0..=4096i32).step_by(8)).into_array();
    let lists = ListArray::try_new(elements, offsets, Validity::NonNullable)?.into_array();
    let strings = VarBinViewArray::from_iter(
        (0..4096).map(|i| Some(format!("products/category/widget-{i:05}"))),
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array();
    let compressor = BtrBlocksCompressorBuilder::from_session(&session)
        .with_compact()
        .build();
    for input in [lists, strings] {
        let compressed = compressor.compress(&input, &mut ctx)?;
        let array_ctx = ArrayContext::empty();
        let mut bytes = ByteBufferMut::empty();
        for buffer in compressed.serialize(&array_ctx, &session, &SerializeOptions::default())? {
            bytes.extend_from_slice(&buffer);
        }
        let ids = array_ctx.to_ids();
        assert!(!ids.contains(&Narrow.id()));
        let decoded = SerializedArray::try_from(bytes.freeze())?.decode(
            input.dtype(),
            input.len(),
            &ReadContext::new(ids),
            &session,
        )?;
        assert_arrays_eq!(decoded, input, &mut ctx);
    }

    Ok(())
}
