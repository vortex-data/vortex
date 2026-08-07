// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The v2 serialized format.
//!
//! Lower parts can be built and computed over freely. What changes with them is the bytes:
//! an array carrying lower parts serializes under `vortex.decimal_byte_parts_v2` rather
//! than the frozen `vortex.decimal_byte_parts` format, so a writer restricted to editions
//! without the v2 format refuses it, and a reader that predates lower parts fails with an
//! unknown-encoding error instead of misreading the children. These tests pin all of that:
//! construction always works, the serialized id tracks the parts, and the permitted-encoding
//! check applies to the serialized id.

#![expect(clippy::tests_outside_test_module)]

use vortex_array::ArrayContext;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArrayVTable;
use vortex_array::IntoArray;
use vortex_array::dtype::DecimalDType;
use vortex_array::serde::SerializeOptions;
use vortex_array::session::ArraySessionExt;
use vortex_buffer::buffer;
use vortex_decimal_byte_parts::DecimalByteParts;
use vortex_decimal_byte_parts::DecimalBytePartsV2;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

fn msp() -> ArrayRef {
    buffer![1i64, 2, 3].into_array()
}

fn lower_part() -> ArrayRef {
    buffer![1u64, 2, 3].into_array()
}

fn session() -> VortexSession {
    let session = vortex_array::array_session();
    vortex_decimal_byte_parts::initialize(&session);
    session
}

/// A single-child array is the stable shape and is always constructible.
#[test]
fn single_child_is_always_allowed() {
    assert!(DecimalByteParts::try_new(msp(), DecimalDType::new(19, 2)).is_ok());
    assert!(
        DecimalByteParts::try_new_with_lower_parts(msp(), vec![], DecimalDType::new(19, 2)).is_ok()
    );
}

/// Building lower parts in memory is always allowed — reading a file requires it. What
/// changes is the serialized format, not what can be constructed.
#[test]
fn lower_parts_can_always_be_constructed() {
    assert!(
        DecimalByteParts::try_new_with_lower_parts(
            msp(),
            vec![lower_part()],
            DecimalDType::new(38, 2),
        )
        .is_ok()
    );
}

/// A single-child array keeps the frozen format id, byte-compatible with every reader since
/// the format froze; lower parts move the array onto the v2 format id.
#[test]
fn serialized_id_tracks_lower_parts() -> VortexResult<()> {
    let session = session();

    let flat = DecimalByteParts::try_new(msp(), DecimalDType::new(19, 2))?.into_array();
    assert_eq!(
        session.array_serialized_id(&flat)?,
        ArrayVTable::id(&DecimalByteParts)
    );

    let wide = DecimalByteParts::try_new_with_lower_parts(
        msp(),
        vec![lower_part()],
        DecimalDType::new(38, 2),
    )?
    .into_array();
    assert_eq!(
        session.array_serialized_id(&wide)?,
        ArrayPlugin::id(&DecimalBytePartsV2)
    );

    Ok(())
}

/// The permitted-encoding check applies to the serialized id. A context restricted to the
/// frozen format — a writer whose enabled editions predate the v2 format — must refuse an
/// array carrying lower parts, however it was obtained.
///
/// `ArrayParts` is public and `DecimalBytePartsData` is a public unit struct, so a caller can
/// assemble slots by hand and go straight to `Array::try_from_parts`, bypassing
/// `try_new_with_lower_parts` entirely. That back door is left open on purpose — it is the
/// same path `deserialize` uses. What must hold is that the resulting array cannot become
/// bytes under the frozen id.
#[test]
fn wide_format_is_refused_where_not_permitted() -> VortexResult<()> {
    use vortex_array::Array;
    use vortex_array::ArrayParts;
    use vortex_array::ArraySlots;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_decimal_byte_parts::DecimalBytePartsData;

    let session = session();

    let mut slots = ArraySlots::with_capacity(2);
    slots.push(Some(msp()));
    slots.push(Some(lower_part()));

    // Assembling the array by hand succeeds: this is the shape a file read produces.
    let array = Array::try_from_parts(
        ArrayParts::new(
            DecimalByteParts,
            DType::Decimal(DecimalDType::new(38, 2), Nullability::NonNullable),
            3,
            DecimalBytePartsData,
        )
        .with_slots(slots),
    )?
    .into_array();
    assert_eq!(array.nchildren(), 2, "expected two limbs");

    // A context permitting only the frozen format refuses to write it.
    let restricted = ArrayContext::empty()
        .with_allowed_ids([ArrayVTable::id(&DecimalByteParts)].into_iter().collect());
    let err = array
        .serialize(&restricted, &session, &SerializeOptions::default())
        .expect_err("expected the permitted-encoding check to refuse the v2 format");
    assert!(
        err.to_string().contains("not permitted"),
        "error should name the permitted-encoding check, got: {err}"
    );

    // Permitting the v2 format id is exactly what allows the same array through.
    let permissive = ArrayContext::empty().with_allowed_ids(
        [
            ArrayVTable::id(&DecimalByteParts),
            ArrayPlugin::id(&DecimalBytePartsV2),
            ArrayVTable::id(&vortex_array::arrays::Primitive),
        ]
        .into_iter()
        .collect(),
    );
    let serialized = array.serialize(&permissive, &session, &SerializeOptions::default())?;
    assert!(!serialized.is_empty());
    assert!(
        permissive
            .to_ids()
            .contains(&ArrayPlugin::id(&DecimalBytePartsV2)),
        "the file's encoding table must carry the v2 format id"
    );

    Ok(())
}
