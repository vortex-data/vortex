// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Executing DateTimeParts over compressed days, seconds and subseconds, such as FoR over
//! BitPacked days.
//!
//! Run with `cargo bench -p vortex-datetime-parts --bench date_time_parts_decode`.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::scalar::Scalar;
use vortex_array::session::ArraySession;
use vortex_datetime_parts::DateTimeParts;
use vortex_datetime_parts::DateTimePartsArraySlotsExt;
use vortex_fastlanes::FoR;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_session::VortexSession;
use vortex_sparse::Sparse;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty().with::<ArraySession>();
    vortex_fastlanes::initialize(&session);
    vortex_datetime_parts::initialize(&session);
    vortex_sparse::initialize(&session);
    session
});

/// A scan split's rows, whose intermediates fit in L2.
const LEN: usize = 64 * 1024;

/// Microsecond timestamps over about two years, with their days as FoR over BitPacked.
fn date_time_parts() -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter((0..LEN as i64).map(|i| i * 997_000_123 + (i % 97)));
    let temporal = TemporalArray::new_timestamp(values.into_array(), TimeUnit::Microseconds, None);
    let parts = DateTimeParts::try_from_temporal(temporal, &mut ctx).unwrap();
    let days = parts
        .days()
        .clone()
        .cast(DType::Primitive(PType::I64, Nullability::NonNullable))
        .unwrap()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    let min = days.as_slice::<i64>().iter().copied().min().unwrap();
    let shifted = PrimitiveArray::from_iter(days.as_slice::<i64>().iter().map(|&d| d - min));
    let bp = bitpack_encode(&shifted, 10, None, &mut ctx).unwrap();
    let days = FoR::try_new(bp.into_array(), Scalar::from(min))
        .unwrap()
        .into_array();
    let seconds = parts
        .seconds()
        .clone()
        .execute::<Canonical>(&mut ctx)
        .unwrap();
    let subseconds = parts
        .subseconds()
        .clone()
        .execute::<Canonical>(&mut ctx)
        .unwrap();
    DateTimeParts::try_new(
        parts.dtype().clone(),
        days,
        seconds.into_array(),
        subseconds.into_array(),
    )
    .unwrap()
    .into_array()
}

#[divan::bench]
fn date_time_parts_execute(bencher: Bencher) {
    let array = date_time_parts();
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| array.execute::<Canonical>(&mut ctx).unwrap());
}

/// Event timestamps with second-level precision and compressed day numbers.
fn bench_timestamps(bencher: Bencher, days: ArrayRef) {
    let len = days.len();
    let dtype =
        DType::Extension(Timestamp::new(TimeUnit::Seconds, days.dtype().nullability()).erased());
    let seconds =
        PrimitiveArray::from_iter((0..len).map(|i| i32::try_from((i * 997) % 86_400).unwrap()));
    let array = DateTimeParts::try_new(
        dtype,
        days,
        seconds.into_array(),
        ConstantArray::new(Scalar::from(0i32), len).into_array(),
    )
    .unwrap()
    .into_array();
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| array.execute::<Canonical>(&mut ctx).unwrap());
}

/// A daily ingestion batch including one late event from the preceding day per hundred rows.
#[divan::bench]
fn timestamps_late_events(bencher: Bencher) {
    let indices = PrimitiveArray::from_iter((0..u32::try_from(LEN).unwrap()).step_by(100));
    let values = PrimitiveArray::from_iter(std::iter::repeat_n(19_999i32, indices.len()));
    let days = Sparse::try_new(
        indices.into_array(),
        values.into_array(),
        LEN,
        Scalar::from(20_000i32),
        &mut SESSION.create_execution_ctx(),
    )
    .unwrap();
    bench_timestamps(bencher, days.into_array());
}

#[divan::bench]
fn intraday_packed_seconds(bencher: Bencher) {
    intraday(bencher, true);
}

#[divan::bench]
fn single_second_packed_subseconds(bencher: Bencher) {
    intraday(bencher, false);
}

fn intraday(bencher: Bencher, pack_seconds: bool) {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype =
        DType::Extension(Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased());
    let days = ConstantArray::new(Scalar::from(20_000i32), LEN).into_array();
    let seconds = PrimitiveArray::from_iter((0..LEN).map(|i| i32::try_from(i / 4).unwrap()));
    let (seconds, subseconds) = if pack_seconds {
        (
            bitpack_encode(&seconds, 14, None, &mut ctx)
                .unwrap()
                .into_array(),
            ConstantArray::new(Scalar::from(0i32), LEN).into_array(),
        )
    } else {
        let subseconds =
            PrimitiveArray::from_iter((0..LEN).map(|i| i32::try_from((i * 251) % 1000).unwrap()));
        (
            ConstantArray::new(Scalar::from(41_563i32), LEN).into_array(),
            bitpack_encode(&subseconds, 10, None, &mut ctx)
                .unwrap()
                .into_array(),
        )
    };
    let array = DateTimeParts::try_new(dtype, days, seconds, subseconds)
        .unwrap()
        .into_array();
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| array.execute::<Canonical>(&mut ctx).unwrap());
}
