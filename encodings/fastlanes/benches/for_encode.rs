// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare `FoR::encode`, which subtracts one reference from every value, against
//! `FoR::encode_chunked`, which subtracts a reference per 1024-element chunk.
//!
//! Run with `cargo bench -p vortex-fastlanes --bench for_encode`.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_fastlanes::FoR;
use vortex_fastlanes::FoRArray;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    session
});

const LENS: &[usize] = &[1024, 16 * 1024];

type Encode = fn(PrimitiveArray, &mut ExecutionCtx) -> VortexResult<FoRArray>;

/// Values that step up by 1000 every chunk, with a small spread within each chunk.
fn values<T: NativePType + TryFrom<usize>>(len: usize) -> Buffer<T> {
    (0..len)
        .map(|i| T::try_from((i / 1024) * 1000 + (i * 7919) % 100).ok().unwrap())
        .collect()
}

fn validity(len: usize, nullable: bool) -> Validity {
    if nullable {
        Validity::from_iter((0..len).map(|i| i % 10 != 0))
    } else {
        Validity::NonNullable
    }
}

fn run<T: NativePType + TryFrom<usize>>(
    bencher: Bencher,
    len: usize,
    nullable: bool,
    encode: Encode,
) {
    let buffer = values::<T>(len);
    let validity = validity(len, nullable);
    bencher
        .counter(ItemsCount::new(len))
        // A fresh array per iteration, so `FoR::encode` can't reuse a cached `Min` statistic.
        .with_inputs(|| PrimitiveArray::new(buffer.clone(), validity.clone()))
        .bench_local_values(|array| encode(array, &mut SESSION.create_execution_ctx()).unwrap());
}

#[divan::bench(types = [u32, i64], args = LENS)]
fn encode<T: NativePType + TryFrom<usize>>(bencher: Bencher, len: usize) {
    run::<T>(bencher, len, false, FoR::encode);
}

#[divan::bench(types = [u32, i64], args = LENS)]
fn encode_chunked<T: NativePType + TryFrom<usize>>(bencher: Bencher, len: usize) {
    run::<T>(bencher, len, false, FoR::encode_chunked);
}

#[divan::bench(types = [u32, i64], args = LENS)]
fn encode_nullable<T: NativePType + TryFrom<usize>>(bencher: Bencher, len: usize) {
    run::<T>(bencher, len, true, FoR::encode);
}

#[divan::bench(types = [u32, i64], args = LENS)]
fn encode_chunked_nullable<T: NativePType + TryFrom<usize>>(bencher: Bencher, len: usize) {
    run::<T>(bencher, len, true, FoR::encode_chunked);
}
