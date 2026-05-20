// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Patched-decode benchmark: chunk-local patch overlay vs full-buffer scatter.
//!
//! Compares decoding a bit-packed primitive column with exceptions ("patches") two ways:
//!
//! - `bitpacked_patches_canonical`: the canonical executor path for a `BitPacked` array
//!   that carries its patches internally. It bit-unpacks the entire column into an
//!   N-element buffer, then scatters the exception values into that buffer by index.
//!   The scatter is a sequence of random writes; once N spills L2/L3 each write can miss.
//!
//! - `bitpacked_patches_chunked`: the same logical data decoded through the chunked engine
//!   as a `PatchedProducer` over a `BitPackedPrimitiveProducer`. Each 1024-element base
//!   chunk is bit-unpacked into the scratch, the patches that fall in that chunk's range
//!   are overlaid while the chunk is still hot in L1, then flushed. The patch writes never
//!   touch a cold cache line.
//!
//! Run with `cargo bench -p vortex-runend --bench chunked_patched`.

use std::fmt;
use std::sync::LazyLock;

use divan::Bencher;
use divan::black_box;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::_chunked_exec::Scratch;
use vortex_array::_chunked_exec::primitive::PrimitiveChunkProducer;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_fastlanes::BitPackedArray;
use vortex_fastlanes::BitPackedArrayExt;
use vortex_fastlanes::BitPackedData;
use vortex_fastlanes::_chunked_exec::build_chunked_patched_over_bitpacked;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let s = VortexSession::empty().with::<ArraySession>();
    vortex_fastlanes::initialize(&s);
    s
});

#[derive(Copy, Clone)]
struct Args {
    len: usize,
    /// Fraction of values that are exceptions, as 1-in-`patch_stride`.
    patch_stride: usize,
    bit_width: u8,
}

impl fmt::Display for Args {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pct = 100.0 / self.patch_stride as f64;
        write!(
            f,
            "len={} patches={:.1}% bw={}",
            self.len, pct, self.bit_width
        )
    }
}

const ARGS: &[Args] = &[
    // 1M: output 4 MiB (2x L2). Patch density swept.
    Args { len: 1_048_576, patch_stride: 100, bit_width: 8 }, // 1%
    Args { len: 1_048_576, patch_stride: 20, bit_width: 8 },  // 5%
    Args { len: 1_048_576, patch_stride: 10, bit_width: 8 },  // 10%
    // 4M: output 16 MiB (8x L2, in L3).
    Args { len: 4_194_304, patch_stride: 100, bit_width: 8 },
    Args { len: 4_194_304, patch_stride: 20, bit_width: 8 },
    Args { len: 4_194_304, patch_stride: 10, bit_width: 8 },
    // 16M: output 64 MiB (deep in L3).
    Args { len: 16_777_216, patch_stride: 100, bit_width: 8 },
    Args { len: 16_777_216, patch_stride: 20, bit_width: 8 },
    Args { len: 16_777_216, patch_stride: 10, bit_width: 8 },
];

/// Build a `BitPacked<u32>` whose values mostly fit in `bit_width` bits, with every
/// `patch_stride`-th value an exception (forcing it into the patches sidecar).
fn make_bitpacked_with_patches(args: Args) -> BitPackedArray {
    let cap = (1u32 << args.bit_width) - 1;
    let values: Vec<u32> = (0..args.len)
        .map(|i| {
            if i % args.patch_stride == 0 {
                // Exception value above the bit-width ceiling.
                cap + 1000 + (i as u32 & 0xffff)
            } else {
                (i as u32) & cap
            }
        })
        .collect();
    let prim = PrimitiveArray::new(Buffer::<u32>::from_iter(values), Validity::NonNullable);
    let mut ctx = SESSION.create_execution_ctx();
    let bp = BitPackedData::encode(&prim.into_array(), args.bit_width, &mut ctx)
        .expect("bitpack encode");
    assert!(bp.patches().is_some(), "expected patches at this density");
    bp
}

/// Layout A: patches stored inside the BitPacked array, decoded canonically.
/// Bit-unpack the whole column, then scatter patches into the full N-element buffer.
#[divan::bench(args = ARGS)]
fn bitpacked_patches_canonical(bencher: Bencher, args: Args) {
    let bp = make_bitpacked_with_patches(args);
    bencher
        .with_inputs(|| bp.clone().into_array())
        .bench_local_refs(|a| {
            let mut ctx = SESSION.create_execution_ctx();
            black_box(a.clone().execute::<PrimitiveArray>(&mut ctx).unwrap())
        });
}

/// Layout B: same data decoded as a chunked PatchedProducer over a patchless
/// BitPackedPrimitiveProducer. Patch overlay happens chunk-locally in L1.
#[divan::bench(args = ARGS)]
fn bitpacked_patches_chunked(bencher: Bencher, args: Args) {
    let bp = make_bitpacked_with_patches(args);
    bencher.with_inputs(|| bp.clone()).bench_local_refs(|bp| {
        let mut ctx = SESSION.create_execution_ctx();
        let mut producer = build_chunked_patched_over_bitpacked::<u32>(bp, &mut ctx)
            .unwrap()
            .expect("non-sliced");
        let mut out = BufferMut::<u32>::with_capacity(args.len);
        let mut scratch = Scratch::<u32>::new();
        let mut written = 0usize;
        while let Some(chunk) = producer.next_chunk(&mut scratch).unwrap() {
            // SAFETY: out has capacity args.len; chunk total never exceeds it.
            unsafe {
                let dst = out.spare_capacity_mut().as_mut_ptr().add(written).cast::<u32>();
                std::ptr::copy_nonoverlapping(chunk.as_ptr(), dst, chunk.len());
            }
            written += chunk.len();
        }
        unsafe { out.set_len(written) };
        black_box(out.freeze())
    });
}
