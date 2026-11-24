// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decoding DateTimeParts whose days are FoR over BitPacked, level by level with `execute`, and
//! block by block with a program: FoR's fused unpack of the days as the leaf, and DateTimeParts'
//! combine of each block of days with its time parts as the op.
//!
//! Run with `cargo bench -p vortex-datetime-parts --bench block_fusion`.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::block::BlockProgram;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::scalar::Scalar;
use vortex_array::session::ArraySession;
use vortex_datetime_parts::DateTimeParts;
use vortex_datetime_parts::DateTimePartsArraySlotsExt;
use vortex_datetime_parts::block::days_block_op;
use vortex_fastlanes::FoR;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_fastlanes::block::for_block_source;
use vortex_session::VortexSession;

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

fn run_program(array: &ArrayRef, ctx: &mut ExecutionCtx) -> PrimitiveArray {
    let parts = array.as_::<DateTimeParts>();
    let days = parts.days();
    let source = for_block_source(days.as_::<FoR>(), ctx).unwrap().unwrap();
    let op = days_block_op(
        array.dtype(),
        days.dtype(),
        parts.seconds(),
        parts.subseconds(),
    )
    .unwrap();
    let mut program = BlockProgram::new(source, days.dtype().as_ptype(), days.len());
    program.push(op, PType::I64);
    let validity = days
        .validity()
        .unwrap()
        .execute_mask(days.len(), ctx)
        .unwrap();
    program
        .run(&validity, array.dtype().nullability(), ctx)
        .unwrap()
}

#[divan::bench]
fn date_time_parts_execute(bencher: Bencher) {
    let array = date_time_parts();
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| array.execute::<Canonical>(&mut ctx).unwrap());
}

#[divan::bench]
fn date_time_parts_program(bencher: Bencher) {
    let array = date_time_parts();
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| run_program(&array, &mut ctx));
}
