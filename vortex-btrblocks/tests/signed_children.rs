// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serialization of compressed internal children that retain their signedness.

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::List;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::Widen;
use vortex_array::arrays::list::ListArraySlotsExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::PType;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
#[cfg(feature = "pco")]
use vortex_array::session::ArraySessionExt;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_buffer::ByteBufferMut;
use vortex_edition::EDITION_DECLARATIONS;
use vortex_edition::EDITION_FAMILIES;
use vortex_edition::EditionSession;
use vortex_edition::EditionSessionExt;
use vortex_edition::declarations::core::CORE_2026_08_3;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty().with::<EditionSession>();
    vortex_alp::initialize(&session);
    vortex_datetime_parts::initialize(&session);
    vortex_decimal_byte_parts::initialize(&session);
    vortex_fastlanes::initialize(&session);
    vortex_fsst::initialize(&session);
    vortex_onpair::initialize(&session);
    vortex_sequence::initialize(&session);
    vortex_sparse::initialize(&session);
    vortex_zigzag::initialize(&session);
    #[cfg(feature = "pco")]
    session.arrays().register(vortex_pco::Pco);
    for family in EDITION_FAMILIES {
        session
            .editions()
            .declare_family(family)
            .vortex_expect("edition families are distinct");
    }
    for declaration in EDITION_DECLARATIONS {
        session
            .register_edition(declaration)
            .vortex_expect("edition dependencies are declared");
    }
    session
        .enable_edition(CORE_2026_08_3)
        .vortex_expect("core edition is registered");
    session
});

fn lists() -> VortexResult<ArrayRef> {
    let elements = PrimitiveArray::from_iter((0..4096i32).map(|i| i % 7)).into_array();
    let offsets = PrimitiveArray::from_iter((0..=4096i32).step_by(8)).into_array();
    Ok(ListArray::try_new(
        elements,
        offsets,
        Validity::NonNullable,
        &mut SESSION.create_execution_ctx(),
    )?
    .into_array())
}

#[rstest]
#[case::lists(lists())]
#[case::strings(Ok(VarBinViewArray::from_iter_str(
    (0..4096).map(|i| format!("products/category/widget-{i:05}"))
).into_array()))]
fn test_internal_signed_buffers_roundtrip(
    #[case] input: VortexResult<ArrayRef>,
) -> VortexResult<()> {
    let input = input?;
    let mut ctx = SESSION.create_execution_ctx();
    let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .with_compact()
        .build();
    let compressed = compressor.compress(&input, &mut ctx)?;
    if input.is::<List>() {
        assert_eq!(
            compressed.as_::<List>().offsets().dtype().as_ptype(),
            PType::I16
        );
    }

    let array_ctx = ArrayContext::empty();
    let mut bytes = ByteBufferMut::empty();
    for buffer in compressed.serialize(&array_ctx, &SESSION, &SerializeOptions::default())? {
        bytes.extend_from_slice(&buffer);
    }
    let ids = array_ctx.to_ids();
    assert!(!ids.contains(&Widen.id()));
    let decoded = SerializedArray::try_from(bytes.freeze())?.decode(
        input.dtype(),
        input.len(),
        &ReadContext::new(ids),
        &SESSION,
    )?;
    if input.is::<List>() {
        assert_eq!(
            decoded.as_::<List>().offsets().dtype().as_ptype(),
            PType::I16
        );
    }
    assert_arrays_eq!(decoded, input, &mut ctx);

    Ok(())
}
