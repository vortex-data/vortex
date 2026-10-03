// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compares `VarBinViewArray` and `VarBinArray` UTF-8 validation across string shapes and null
//! densities.

#![allow(clippy::unwrap_used)]

use std::sync::Arc;
use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

const N: usize = 1 << 16;
static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

/// `n` strings of `len` bytes, concatenated, plus their offsets.
fn strings(n: usize, len: usize, multibyte: bool) -> (Vec<u8>, Vec<u32>) {
    let unit = if multibyte {
        "zażółć gęślą jaźń "
    } else {
        "abcdefghijklmnopqrstuvwxyz"
    };
    let mut bytes = Vec::new();
    let mut offsets = vec![0u32];
    for i in 0..n {
        let s: String = unit.chars().cycle().skip(i % 7).take(len).collect();
        bytes.extend_from_slice(s.as_bytes());
        offsets.push(u32::try_from(bytes.len()).unwrap());
    }
    (bytes, offsets)
}

fn views(bytes: &[u8], offsets: &[u32], count: usize) -> Vec<BinaryView> {
    offsets
        .windows(2)
        .take(count)
        .map(|o| BinaryView::make_view(&bytes[o[0] as usize..o[1] as usize], 0, o[0]))
        .collect()
}

/// `null_every == 0` means non-nullable.
fn validity(n: usize, null_every: usize) -> Validity {
    if null_every == 0 {
        Validity::NonNullable
    } else {
        Validity::from_bit_buffer(
            BitBuffer::from_iter((0..n).map(|i| i % null_every != 0)),
            Nullability::Nullable,
        )
    }
}

fn nullability(null_every: usize) -> Nullability {
    if null_every == 0 {
        Nullability::NonNullable
    } else {
        Nullability::Nullable
    }
}

fn run_vbv(bencher: Bencher, len: usize, multibyte: bool, null_every: usize, count: usize) {
    let (bytes, offsets) = strings(N, len, multibyte);
    let v = views(&bytes, &offsets, count);
    let buffers: Arc<[ByteBuffer]> = Arc::new([ByteBuffer::from(bytes)]);
    let dtype = DType::Utf8(nullability(null_every));
    let validity = validity(count, null_every);
    bencher
        .with_inputs(|| {
            (
                Buffer::copy_from(&v),
                Arc::clone(&buffers),
                dtype.clone(),
                validity.clone(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(views, buffers, dtype, validity, mut ctx)| {
            VarBinViewArray::try_new(views, buffers, dtype, validity, &mut ctx).unwrap()
        });
}

#[divan::bench(args = [0, 2, 10])]
fn vbv_short20(bencher: Bencher, null_every: usize) {
    run_vbv(bencher, 20, false, null_every, N);
}

#[divan::bench(args = [0, 2, 10])]
fn vbv_long200(bencher: Bencher, null_every: usize) {
    run_vbv(bencher, 200, false, null_every, N);
}

#[divan::bench(args = [0, 2])]
fn vbv_multibyte24(bencher: Bencher, null_every: usize) {
    run_vbv(bencher, 24, true, null_every, N);
}

/// 1024 views into the buffer of 65536 strings, as after a slice.
#[divan::bench(args = [0, 2])]
fn vbv_sliced_short20(bencher: Bencher, null_every: usize) {
    run_vbv(bencher, 20, false, null_every, 1024);
}

fn run_varbin(bencher: Bencher, len: usize, null_every: usize) {
    let (bytes, offsets) = strings(N, len, false);
    let offsets: Vec<i32> = offsets
        .into_iter()
        .map(|o| i32::try_from(o).unwrap())
        .collect();
    let bytes = ByteBuffer::from(bytes);
    let offsets = Buffer::from(offsets).into_array();
    let dtype = DType::Utf8(nullability(null_every));
    let validity = validity(N, null_every);
    bencher
        .with_inputs(|| {
            (
                offsets.clone(),
                bytes.clone(),
                dtype.clone(),
                validity.clone(),
            )
        })
        .bench_values(|(offsets, bytes, dtype, validity)| {
            VarBinArray::try_new(offsets, bytes, dtype, validity).unwrap()
        });
}

#[divan::bench(args = [0, 2, 10])]
fn varbin_short20(bencher: Bencher, null_every: usize) {
    run_varbin(bencher, 20, null_every);
}
