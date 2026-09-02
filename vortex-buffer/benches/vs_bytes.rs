// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Head-to-head microbenchmarks against `bytes`, the crate `vortex-buffer` used to be built on,
//! and against `arrow_buffer::Buffer`, the other aligned buffer Vortex interoperates with.
//!
//! `arrow` is the closest comparison in kind: like `vortex`, it over-aligns (to 128 bytes on
//! x86_64, rounding capacity up to a multiple of 64) and can hand an adopted `Vec` back out. It
//! differs in allocating an `Arc<Bytes>` for every frozen buffer, which is the cost the tagged
//! state word in `vortex-bytes` exists to avoid.
//!
//! Two `vortex` variants are measured wherever alignment is in play, because the comparison is
//! otherwise unfair in our favour and then unfair against us:
//!
//! * `vortex` uses the crate default, over-aligning to [`Alignment::DEFAULT_ALIGNMENT`] (256) for
//!   SIMD and CUDA. This is what Vortex actually allocates, and it is strictly more work than
//!   `bytes` does.
//! * `vortex_unaligned` asks for no over-alignment, which is the like-for-like comparison with
//!   `bytes::BytesMut`.

use arrow_buffer::Buffer as ArrowBuffer;
use arrow_buffer::MutableBuffer as ArrowBufferMut;
use bytes::Buf;
use bytes::BufMut;
use bytes::Bytes;
use bytes::BytesMut;
use divan::Bencher;
use divan::black_box;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[64, 1024, 65_536];

#[expect(
    clippy::cast_possible_truncation,
    reason = "truncating to the low byte is the point of the pattern"
)]
fn payload(n: usize) -> Vec<u8> {
    (0..n).map(|i| i as u8).collect()
}

fn unaligned(capacity: usize) -> ByteBufferMut {
    ByteBufferMut::with_capacity_preferred_aligned(capacity, Alignment::none(), None)
}

// ---------------------------------------------------------------------------------------------
// Allocate, fill, freeze: the path every buffer Vortex builds goes through.
// ---------------------------------------------------------------------------------------------

#[divan::bench(args = SIZES)]
fn build_bytes(bencher: Bencher, n: usize) {
    let src = payload(n);
    bencher.bench(|| {
        let mut b = BytesMut::with_capacity(n);
        b.put_slice(black_box(&src));
        black_box(b.freeze())
    });
}

#[divan::bench(args = SIZES)]
fn build_arrow(bencher: Bencher, n: usize) {
    let src = payload(n);
    bencher.bench(|| {
        let mut b = ArrowBufferMut::with_capacity(n);
        b.extend_from_slice(black_box(&src));
        black_box(ArrowBuffer::from(b))
    });
}

#[divan::bench(args = SIZES)]
fn build_vortex(bencher: Bencher, n: usize) {
    let src = payload(n);
    bencher.bench(|| {
        let mut b = ByteBufferMut::with_capacity(n);
        b.extend_from_slice(black_box(&src));
        black_box(b.freeze())
    });
}

#[divan::bench(args = SIZES)]
fn build_vortex_unaligned(bencher: Bencher, n: usize) {
    let src = payload(n);
    bencher.bench(|| {
        let mut b = unaligned(n);
        b.extend_from_slice(black_box(&src));
        black_box(b.freeze())
    });
}

/// How `vortex-buffer` produced a 256-aligned buffer when it was built on `bytes`: allocate
/// `n + alignment` bytes and walk the pointer forward to the next aligned offset. This is the
/// arm the new implementation actually replaced, so it is the one to compare `build_vortex`
/// against - `build_bytes` buys no alignment at all.
#[divan::bench(args = SIZES)]
fn build_vortex_as_before(bencher: Bencher, n: usize) {
    let src = payload(n);
    let alignment = *Alignment::DEFAULT_ALIGNMENT;
    bencher.bench(|| {
        let mut b = BytesMut::with_capacity(n + alignment);
        let padding = b.as_ptr().align_offset(alignment);
        // SAFETY: the buffer is empty and has at least `padding` bytes of capacity.
        unsafe { b.set_len(padding) };
        drop(b.split_to(padding));
        b.put_slice(black_box(&src));
        black_box(b.freeze())
    });
}

// ---------------------------------------------------------------------------------------------
// Growth from empty: repeated reallocation, where the alignment strategy shows up most.
// ---------------------------------------------------------------------------------------------

#[divan::bench(args = SIZES)]
fn grow_bytes(bencher: Bencher, n: usize) {
    let chunk = payload(64);
    bencher.bench(|| {
        let mut b = BytesMut::new();
        while b.len() < n {
            b.put_slice(black_box(&chunk));
        }
        black_box(b.freeze())
    });
}

#[divan::bench(args = SIZES)]
fn grow_arrow(bencher: Bencher, n: usize) {
    let chunk = payload(64);
    bencher.bench(|| {
        let mut b = ArrowBufferMut::new(0);
        while b.len() < n {
            b.extend_from_slice(black_box(&chunk));
        }
        black_box(ArrowBuffer::from(b))
    });
}

#[divan::bench(args = SIZES)]
fn grow_vortex(bencher: Bencher, n: usize) {
    let chunk = payload(64);
    bencher.bench(|| {
        let mut b = ByteBufferMut::empty();
        while b.len() < n {
            b.extend_from_slice(black_box(&chunk));
        }
        black_box(b.freeze())
    });
}

#[divan::bench(args = SIZES)]
fn grow_vortex_unaligned(bencher: Bencher, n: usize) {
    let chunk = payload(64);
    bencher.bench(|| {
        let mut b = unaligned(0);
        while b.len() < n {
            b.extend_from_slice(black_box(&chunk));
        }
        black_box(b.freeze())
    });
}

// ---------------------------------------------------------------------------------------------
// Sharing: clone and slice an already-frozen buffer.
// ---------------------------------------------------------------------------------------------

#[divan::bench]
fn clone_bytes(bencher: Bencher) {
    let b = Bytes::from(payload(4096));
    bencher.bench(|| black_box(black_box(&b).clone()));
}

#[divan::bench]
fn clone_arrow(bencher: Bencher) {
    let b = ArrowBuffer::from_vec(payload(4096));
    bencher.bench(|| black_box(black_box(&b).clone()));
}

#[divan::bench]
fn clone_vortex(bencher: Bencher) {
    let b = ByteBuffer::from(payload(4096));
    bencher.bench(|| black_box(black_box(&b).clone()));
}

#[divan::bench]
fn slice_bytes(bencher: Bencher) {
    let b = Bytes::from(payload(4096));
    bencher.bench(|| black_box(black_box(&b).slice(64..1088)));
}

#[divan::bench]
fn slice_arrow(bencher: Bencher) {
    let b = ArrowBuffer::from_vec(payload(4096));
    bencher.bench(|| black_box(black_box(&b).slice_with_length(64, 1024)));
}

#[divan::bench]
fn slice_vortex(bencher: Bencher) {
    let b = ByteBuffer::from(payload(4096));
    bencher.bench(|| black_box(black_box(&b).slice(64..1088)));
}

#[divan::bench]
fn advance_bytes(bencher: Bencher) {
    let b = Bytes::from(payload(4096));
    bencher.bench(|| {
        let mut b = black_box(&b).clone();
        b.advance(64);
        black_box(b)
    });
}

/// `arrow` has no in-place advance; `slice(offset)` on a clone is the equivalent.
#[divan::bench]
fn advance_arrow(bencher: Bencher) {
    let b = ArrowBuffer::from_vec(payload(4096));
    bencher.bench(|| {
        let b = black_box(&b).clone();
        black_box(b.slice(64))
    });
}

#[divan::bench]
fn advance_vortex(bencher: Bencher) {
    let b = ByteBuffer::from(payload(4096));
    bencher.bench(|| {
        let mut b = black_box(&b).clone();
        b.advance(64);
        black_box(b)
    });
}

// ---------------------------------------------------------------------------------------------
// Reclaiming mutability. This is the capability the rewrite was for: `bytes` can only do it for
// bytes that came out of `BytesMut::freeze`, and never for adopted memory.
// ---------------------------------------------------------------------------------------------

#[divan::bench(args = SIZES)]
fn freeze_thaw_bytes(bencher: Bencher, n: usize) {
    let src = payload(n);
    bencher.bench(|| {
        let mut b = BytesMut::with_capacity(n);
        b.put_slice(black_box(&src));
        let frozen = b.freeze();
        black_box(frozen.try_into_mut().ok())
    });
}

#[divan::bench(args = SIZES)]
fn freeze_thaw_arrow(bencher: Bencher, n: usize) {
    let src = payload(n);
    bencher.bench(|| {
        let mut b = ArrowBufferMut::with_capacity(n);
        b.extend_from_slice(black_box(&src));
        let frozen = ArrowBuffer::from(b);
        black_box(frozen.into_mutable().ok())
    });
}

#[divan::bench(args = SIZES)]
fn freeze_thaw_vortex(bencher: Bencher, n: usize) {
    let src = payload(n);
    bencher.bench(|| {
        let mut b = ByteBufferMut::with_capacity(n);
        b.extend_from_slice(black_box(&src));
        let frozen = b.freeze();
        black_box(frozen.try_into_mut().ok())
    });
}

/// Adopt a `Vec` and take mutability back. `bytes` cannot: `Bytes::from(Vec)` is not promotable,
/// so `try_into_mut` fails and a caller has to copy. Measured here as the copy it forces.
#[divan::bench(args = SIZES)]
fn adopt_vec_then_mutate_bytes(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| payload(size))
        .bench_values(|values: Vec<u8>| {
            let bytes = Bytes::from(values);
            let mutable = bytes
                .try_into_mut()
                .unwrap_or_else(|bytes| BytesMut::from(bytes.as_ref()));
            black_box(mutable)
        });
}

/// `arrow` can do this one: an adopted `Vec` is a `Deallocation::Standard`, so `into_mutable`
/// succeeds. Its cost is the `Arc<Bytes>` that `from_vec` allocates and `into_mutable` frees.
#[divan::bench(args = SIZES)]
fn adopt_vec_then_mutate_arrow(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| payload(size))
        .bench_values(|values: Vec<u8>| {
            let buffer = ArrowBuffer::from_vec(values);
            let mutable = buffer
                .into_mutable()
                .unwrap_or_else(|buffer| ArrowBufferMut::from(buffer.as_slice().to_vec()));
            black_box(mutable)
        });
}

#[divan::bench(args = SIZES)]
fn adopt_vec_then_mutate_vortex(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| payload(size))
        .bench_values(|values: Vec<u8>| {
            let buffer = ByteBuffer::from(values);
            black_box(buffer.into_mut())
        });
}

/// Adoption on its own, to separate it from the hand-back below.
#[divan::bench(args = SIZES)]
fn adopt_vec_bytes(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| payload(size))
        .bench_values(|values: Vec<u8>| black_box(Bytes::from(values)));
}

#[divan::bench(args = SIZES)]
fn adopt_vec_arrow(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| payload(size))
        .bench_values(|values: Vec<u8>| black_box(ArrowBuffer::from_vec(values)));
}

#[divan::bench(args = SIZES)]
fn adopt_vec_vortex(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| payload(size))
        .bench_values(|values: Vec<u8>| black_box(ByteBuffer::from(values)));
}

/// Hand a buffer back out as a `Vec<u8>`.
#[divan::bench(args = SIZES)]
fn into_vec_bytes(bencher: Bencher, n: usize) {
    bencher
        .with_inputs(|| Bytes::from(payload(n)))
        .bench_values(|b: Bytes| black_box(Vec::<u8>::from(b)));
}

#[divan::bench(args = SIZES)]
fn into_vec_arrow(bencher: Bencher, n: usize) {
    bencher
        .with_inputs(|| ArrowBuffer::from_vec(payload(n)))
        .bench_values(|b: ArrowBuffer| {
            black_box(
                b.into_vec::<u8>()
                    .unwrap_or_else(|buffer| buffer.as_slice().to_vec()),
            )
        });
}

#[divan::bench(args = SIZES)]
fn into_vec_vortex(bencher: Bencher, n: usize) {
    bencher
        .with_inputs(|| ByteBuffer::from(payload(n)))
        .bench_values(|b: ByteBuffer| black_box(b.into_vec()));
}
