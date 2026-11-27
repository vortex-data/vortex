// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Temporal scheme refinement by allowed serialized IDs, `Time` compression, and coarse-grained
//! subseconds scaling.

#![cfg(test)]

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Extension;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::assert_arrays_eq;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::session::ArraySessionExt;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::temporal::TemporalScheme;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_datetime_parts::DateTimeParts;
use vortex_datetime_parts::datetime_parts_v1_id;
use vortex_datetime_parts::datetime_parts_v2_id;
use vortex_edition::EDITION_DECLARATIONS;
use vortex_edition::EDITION_FAMILIES;
use vortex_edition::EditionSession;
use vortex_edition::EditionSessionExt;
use vortex_edition::declarations::core::CORE_2026_08_3;
use vortex_edition::declarations::unstable::UNSTABLE;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

static TEMPORAL_V2: TemporalScheme = TemporalScheme::v2();

/// Registers datetime-parts and fastlanes encodings, enabling no editions.
static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    vortex_datetime_parts::initialize(&session);
    vortex_fastlanes::initialize(&session);
    session
});

fn session_with_editions(enable_unstable: bool) -> VortexSession {
    let session = VortexSession::empty().with::<EditionSession>();
    for family in EDITION_FAMILIES {
        session
            .editions()
            .declare_family(family)
            .expect("first-party edition family");
    }
    for declaration in EDITION_DECLARATIONS {
        session
            .register_edition(declaration)
            .expect("first-party edition");
    }
    session
        .enable_edition(CORE_2026_08_3)
        .expect("core edition is registered");
    if enable_unstable {
        session
            .enable_edition(UNSTABLE)
            .expect("unstable edition is registered");
    }
    vortex_datetime_parts::initialize(&session);
    vortex_fastlanes::initialize(&session);
    session
}

static CORE_SESSION: LazyLock<VortexSession> = LazyLock::new(|| session_with_editions(false));
static UNSTABLE_SESSION: LazyLock<VortexSession> = LazyLock::new(|| session_with_editions(true));

fn serde_round_trip(array: &ArrayRef) -> VortexResult<(ArrayId, ArrayRef)> {
    let serialized_id = SESSION
        .array_serialize(array)?
        .ok_or_else(|| vortex_err!("expected serializable array"))?
        .serialized_id;
    let array_ctx = ArrayContext::empty();
    let mut bytes = ByteBufferMut::empty();
    for buffer in array.serialize(&array_ctx, &SESSION, &SerializeOptions::default())? {
        bytes.extend_from_slice(buffer.as_ref());
    }

    let decoded = SerializedArray::try_from(bytes.freeze())?.decode(
        array.dtype(),
        array.len(),
        &ReadContext::new(array_ctx.to_ids()),
        &SESSION,
    )?;
    Ok((serialized_id, decoded))
}

#[derive(Debug, Clone, Copy)]
enum ExpectedMode {
    None,
    V1,
    V2,
}

fn timestamp_ms_in_us() -> ArrayRef {
    // Realistic multi-day microsecond timestamps at millisecond precision (like NYC Taxi):
    // subseconds are all multiples of 1_000, non-monotone across days so DateTimeParts wins.
    let base = 1_700_000_000_000_000i64;
    let values: Buffer<i64> = (0..4096i64)
        .map(|i| {
            let day_offset = ((i * 37) % 30) * 86_400 * 1_000_000;
            let sec_offset = ((i * 7_919) % 86_400) * 1_000_000;
            let ms_offset = ((i * 131) % 1_000) * 1_000;
            base + day_offset + sec_offset + ms_offset
        })
        .collect();
    TemporalArray::new_timestamp(
        PrimitiveArray::new(values, Validity::NonNullable).into_array(),
        TimeUnit::Microseconds,
        Some("UTC".into()),
    )
    .into_array()
}

fn time_us_whole_seconds() -> ArrayRef {
    // Whole-second times of day stored in `vortex.time[µs]` (like appian-credit-card):
    // raw storage needs 37 bits/row; DateTimeParts splits into days=Constant(0), seconds=17 bits,
    // subseconds=Constant(0) -> >2x smaller!
    let values: Buffer<i64> = (0..4096i64)
        .map(|i| ((i * 7_919) % 86_400) * 1_000_000)
        .collect();
    TemporalArray::new_time(
        PrimitiveArray::new(values, Validity::NonNullable).into_array(),
        TimeUnit::Microseconds,
    )
    .into_array()
}

#[rstest]
#[case::no_editions(|| BtrBlocksCompressorBuilder::from_session(&SESSION), ExpectedMode::None)]
#[case::core_edition(|| BtrBlocksCompressorBuilder::from_session(&CORE_SESSION), ExpectedMode::V1)]
#[case::unstable_edition(|| BtrBlocksCompressorBuilder::from_session(&UNSTABLE_SESSION), ExpectedMode::V2)]
#[case::editions_disabled(
    || BtrBlocksCompressorBuilder::from_session(&SESSION).disable_editions(),
    ExpectedMode::V2
)]
#[case::unrestricted(
    || BtrBlocksCompressorBuilder::from_session(&SESSION).unrestricted(),
    ExpectedMode::V2
)]
#[case::cuda(
    || BtrBlocksCompressorBuilder::from_session(&SESSION).unrestricted().only_cuda_compatible(),
    ExpectedMode::V1
)]
fn temporal_scheme_follows_allowed_ids_for_scaled_timestamp(
    #[case] builder: fn() -> BtrBlocksCompressorBuilder,
    #[case] expected: ExpectedMode,
) -> VortexResult<()> {
    let array = timestamp_ms_in_us();
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = builder().build().compress(&array, &mut ctx)?;

    match expected {
        ExpectedMode::None => {
            assert!(
                compressed.is::<Extension>(),
                "{}",
                compressed.display_tree()
            );
        }
        ExpectedMode::V1 => {
            let dtp = compressed.as_opt::<DateTimeParts>().ok_or_else(|| {
                vortex_err!("expected DateTimeParts, got {}", compressed.display_tree())
            })?;
            assert_eq!(dtp.subseconds_divisor(), 1);
            let (id, decoded) = serde_round_trip(&compressed)?;
            assert_eq!(id, datetime_parts_v1_id());
            assert_arrays_eq!(decoded, array, &mut ctx);
        }
        ExpectedMode::V2 => {
            let dtp = compressed.as_opt::<DateTimeParts>().ok_or_else(|| {
                vortex_err!("expected DateTimeParts, got {}", compressed.display_tree())
            })?;
            assert_eq!(dtp.subseconds_divisor(), 1_000);
            let (id, decoded) = serde_round_trip(&compressed)?;
            assert_eq!(id, datetime_parts_v2_id());
            assert_arrays_eq!(decoded, array, &mut ctx);
        }
    }
    assert_arrays_eq!(compressed, array, &mut ctx);
    Ok(())
}

#[rstest]
#[case::no_editions(|| BtrBlocksCompressorBuilder::from_session(&SESSION), ExpectedMode::None)]
#[case::core_edition(|| BtrBlocksCompressorBuilder::from_session(&CORE_SESSION), ExpectedMode::V1)]
#[case::unstable_edition(|| BtrBlocksCompressorBuilder::from_session(&UNSTABLE_SESSION), ExpectedMode::V2)]
#[case::cuda(
    || BtrBlocksCompressorBuilder::from_session(&SESSION).unrestricted().only_cuda_compatible(),
    ExpectedMode::V1
)]
fn temporal_scheme_follows_allowed_ids_for_time(
    #[case] builder: fn() -> BtrBlocksCompressorBuilder,
    #[case] expected: ExpectedMode,
) -> VortexResult<()> {
    let array = time_us_whole_seconds();
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = builder().build().compress(&array, &mut ctx)?;

    match expected {
        ExpectedMode::None | ExpectedMode::V1 => {
            assert!(
                compressed.is::<Extension>(),
                "{}",
                compressed.display_tree()
            );
        }
        ExpectedMode::V2 => {
            let dtp = compressed.as_opt::<DateTimeParts>().ok_or_else(|| {
                vortex_err!("expected DateTimeParts, got {}", compressed.display_tree())
            })?;
            assert_eq!(dtp.subseconds_divisor(), 1);
            let (id, decoded) = serde_round_trip(&compressed)?;
            assert_eq!(id, datetime_parts_v2_id());
            assert_arrays_eq!(decoded, array, &mut ctx);
        }
    }
    assert_arrays_eq!(compressed, array, &mut ctx);
    Ok(())
}

#[test]
fn time_us_whole_seconds_compresses_much_smaller_in_v2() -> VortexResult<()> {
    let array = time_us_whole_seconds();
    let mut ctx = SESSION.create_execution_ctx();
    let v1_compressed = BtrBlocksCompressorBuilder::from_session(&CORE_SESSION)
        .build()
        .compress(&array, &mut ctx)?;
    let v2_compressed = BtrBlocksCompressorBuilder::from_session(&UNSTABLE_SESSION)
        .build()
        .compress(&array, &mut ctx)?;

    assert!(v1_compressed.is::<Extension>());
    assert!(v2_compressed.is::<DateTimeParts>());
    // 37 bits/row vs ~17 bits/row -> v2 must be less than half the size of v1.
    assert!(
        v2_compressed.nbytes() * 2 < v1_compressed.nbytes(),
        "expected v2 ({} B) to be < half of v1 ({} B)",
        v2_compressed.nbytes(),
        v1_compressed.nbytes(),
    );
    assert_arrays_eq!(v2_compressed, array, &mut ctx);
    Ok(())
}

#[test]
fn timestamp_ms_in_us_compresses_smaller_in_v2() -> VortexResult<()> {
    let array = timestamp_ms_in_us();
    let mut ctx = SESSION.create_execution_ctx();
    let v1_compressed = BtrBlocksCompressorBuilder::from_session(&CORE_SESSION)
        .build()
        .compress(&array, &mut ctx)?;
    let v2_compressed = BtrBlocksCompressorBuilder::from_session(&UNSTABLE_SESSION)
        .build()
        .compress(&array, &mut ctx)?;

    assert!(v1_compressed.is::<DateTimeParts>());
    assert!(v2_compressed.is::<DateTimeParts>());
    assert_eq!(v1_compressed.as_::<DateTimeParts>().subseconds_divisor(), 1);
    assert_eq!(
        v2_compressed.as_::<DateTimeParts>().subseconds_divisor(),
        1_000
    );
    // 10-bit scaled BitPacked subseconds beats both 20-bit unscaled BitPacked and IntDict.
    assert!(
        v2_compressed.nbytes() + 2_000 < v1_compressed.nbytes(),
        "expected v2 ({} B) to save >2 KB vs v1 ({} B)",
        v2_compressed.nbytes(),
        v1_compressed.nbytes(),
    );
    assert_arrays_eq!(v2_compressed, array, &mut ctx);
    Ok(())
}

#[rstest]
#[case::time_ms(TimeUnit::Milliseconds, 1)]
#[case::time_us(TimeUnit::Microseconds, 1_000)]
#[case::time_ns_ms(TimeUnit::Nanoseconds, 1_000_000)]
#[case::time_ns_us(TimeUnit::Nanoseconds, 1_000)]
fn nullable_time_with_dirty_null_slots_roundtrips(
    #[case] unit: TimeUnit,
    #[case] expected_divisor: u32,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let validity = Validity::from_iter((0..4096).map(|i| i % 9 != 0));
    let array = match unit {
        TimeUnit::Milliseconds => {
            let values: Buffer<i32> = (0..4096i32)
                .map(|i| {
                    if i % 9 == 0 {
                        // Dirty null slot
                        123_456_789
                    } else {
                        ((i * 7_919) % 86_400) * 1_000
                    }
                })
                .collect();
            TemporalArray::new_time(PrimitiveArray::new(values, validity).into_array(), unit)
                .into_array()
        }
        TimeUnit::Microseconds => {
            let values: Buffer<i64> = (0..4096i64)
                .map(|i| {
                    if i % 9 == 0 {
                        // Dirty non-multiple negative null slot
                        -999_123
                    } else {
                        ((i * 7_919) % 86_400) * 1_000_000 + ((i * 131) % 1_000) * 1_000
                    }
                })
                .collect();
            TemporalArray::new_time(PrimitiveArray::new(values, validity).into_array(), unit)
                .into_array()
        }
        TimeUnit::Nanoseconds => {
            let step = i64::from(expected_divisor);
            let sub_mod = 1_000_000_000 / step;
            let values: Buffer<i64> = (0..4096i64)
                .map(|i| {
                    if i % 9 == 0 {
                        -777_777_777
                    } else {
                        ((i * 7_919) % 86_400) * 1_000_000_000 + ((i * 131) % sub_mod) * step
                    }
                })
                .collect();
            TemporalArray::new_time(PrimitiveArray::new(values, validity).into_array(), unit)
                .into_array()
        }
        _ => unreachable!(),
    };

    let compressed = BtrBlocksCompressorBuilder::from_session(&UNSTABLE_SESSION)
        .build()
        .compress(&array, &mut ctx)?;
    let dtp = compressed
        .as_opt::<DateTimeParts>()
        .ok_or_else(|| vortex_err!("expected DateTimeParts, got {}", compressed.display_tree()))?;
    assert_eq!(dtp.subseconds_divisor(), expected_divisor);

    let (id, decoded) = serde_round_trip(&compressed)?;
    assert_eq!(id, datetime_parts_v2_id());
    assert_arrays_eq!(decoded, array, &mut ctx);
    Ok(())
}

#[test]
fn refinement_does_not_restore_excluded_temporal() -> VortexResult<()> {
    let array = timestamp_ms_in_us();
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .unrestricted()
        .exclude_schemes([TEMPORAL_V2.id()])
        .build()
        .compress(&array, &mut ctx)?;
    assert!(
        compressed.is::<Extension>(),
        "{}",
        compressed.display_tree()
    );
    assert_arrays_eq!(compressed, array, &mut ctx);
    Ok(())
}
