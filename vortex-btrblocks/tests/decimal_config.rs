// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::ArrayId;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::DecimalArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DecimalDType;
use vortex_array::session::ArraySessionExt;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::decimal::DecimalScheme;
use vortex_buffer::Buffer;
use vortex_decimal_byte_parts::DecimalByteParts;
use vortex_decimal_byte_parts::decimal_byte_parts_v1_id;
use vortex_decimal_byte_parts::decimal_byte_parts_v2_id;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_set::HashSet;

#[rstest]
#[case::neither(vec![], false)]
#[case::v1(vec![decimal_byte_parts_v1_id()], true)]
#[case::v2_only(vec![decimal_byte_parts_v2_id()], false)]
#[case::both(vec![decimal_byte_parts_v1_id(), decimal_byte_parts_v2_id()], true)]
fn supported_ids(#[case] ids: Vec<ArrayId>, #[case] supported: bool) {
    let allowed_serialized_ids = ids.into_iter().collect();
    // Construction is unconditional; the compressor independently gates the scheme.
    let scheme = DecimalScheme::new(Some(&allowed_serialized_ids));
    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(DecimalScheme::new)
        .retain_allowed_encodings(&allowed_serialized_ids)
        .build();
    assert_eq!(compressor.has_scheme(scheme.id()), supported);
}

#[rstest]
fn decimal_format_follows_configuration(#[values(false, true)] wide: bool) -> VortexResult<()> {
    let session = vortex_array::array_session();
    vortex_decimal_byte_parts::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let base = if wide { 1i128 << 70 } else { 0 };
    let array = DecimalArray::new(
        (0..128i128).map(|i| base + i).collect::<Buffer<i128>>(),
        DecimalDType::new(38, 2),
        Validity::NonNullable,
    )
    .into_array();
    let builder = BtrBlocksCompressorBuilder::empty().with_new_scheme(DecimalScheme::new);
    let v1 = builder
        .clone()
        .retain_allowed_encodings(&HashSet::from([decimal_byte_parts_v1_id()]))
        .build();
    let v2 = builder
        .retain_allowed_encodings(&HashSet::from([
            decimal_byte_parts_v1_id(),
            decimal_byte_parts_v2_id(),
        ]))
        .build();

    for (compressor, allow_v2) in [(&v1, false), (&v2, true), (&v1, false)] {
        let compressed = compressor.compress(&array, &mut ctx)?;
        assert_eq!(compressed.is::<DecimalByteParts>(), !wide || allow_v2);
        if compressed.is::<DecimalByteParts>() {
            let serialized = session
                .array_serialize(&compressed)?
                .ok_or_else(|| vortex_err!("expected serializable decimal byte parts"))?;
            let expected = if wide {
                decimal_byte_parts_v2_id()
            } else {
                decimal_byte_parts_v1_id()
            };
            assert_eq!(serialized.serialized_id, expected);
        }
        assert_arrays_eq!(array, compressed, &mut ctx);
    }
    Ok(())
}
