// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compares decoding FastLanes data through a [`BlockProgram`] with the fused decoders.
//!
//! The program's source unpacks each 1024-value chunk into its scratch block, and a FoR op adds
//! the chunk's reference while writing the output. `direct_*` unpacks straight into the output
//! and adds the reference there, as a source that writes the output itself would. `fused_*`
//! decodes the same data as a vortex array, through the fused unpack-and-add.
//!
//! Run with `cargo bench -p vortex-fastlanes --bench block_vs_fused`.

#![expect(clippy::unwrap_used)]

use std::ops::Range;
use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::BytesCount;
use fastlanes::BitPacking;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::block::Block;
use vortex_array::block::BlockBuf;
use vortex_array::block::BlockOp;
use vortex_array::block::BlockProgram;
use vortex_array::block::BlockSource;
use vortex_array::block::BlockValidity;
use vortex_array::block::SourceBlock;
use vortex_array::block::Written;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::session::ArraySession;
use vortex_array::test_harness::mark_optimized;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_compute::prefetch::prefetch_write;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::FoR;
use vortex_fastlanes::FoRArrayExt;
use vortex_fastlanes::FoRArraySlotsExt;
use vortex_fastlanes::FoRReferences;
use vortex_mask::Mask;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<KernelSession>()
        .with::<AggregateFnSession>()
        .with::<MemorySession>();
    vortex_fastlanes::initialize(&session);
    session
});

/// Bytes of decoded values: L2-resident, L2-sized, and larger than the last-level cache.
const OUTPUT_BYTES: &[usize] = &[64 * 1024, 512 * 1024, 8 * 1024 * 1024];

const CHUNK: usize = 1024;
const BIT_WIDTH: usize = 7;
/// `u32`s per packed chunk.
const PACKED: usize = CHUNK * BIT_WIDTH / 32;

/// The decoded values, a multiple of a chunk long: each chunk climbs by 1000 above its own
/// reference.
fn values(len: usize) -> Buffer<u32> {
    (0..len)
        .map(|i| u32::try_from(1000 * (1 + i / CHUNK) + (i * 7919) % 100).unwrap())
        .collect()
}

/// Packed chunks of `values` less their chunk's minimum, and those minimums.
struct Packed {
    packed: Buffer<u32>,
    references: Buffer<u32>,
    len: usize,
}

impl Packed {
    fn new(values: &[u32]) -> Self {
        assert_eq!(values.len() % CHUNK, 0);
        let mut packed = BufferMut::<u32>::zeroed(values.len() / CHUNK * PACKED);
        let mut references = BufferMut::<u32>::with_capacity(values.len() / CHUNK);
        for (chunk, out) in values
            .chunks_exact(CHUNK)
            .zip(packed.chunks_exact_mut(PACKED))
        {
            let reference = *chunk.iter().min().unwrap();
            let encoded: Vec<u32> = chunk.iter().map(|v| v - reference).collect();
            // SAFETY: `encoded` is one chunk and `out` holds one chunk at `BIT_WIDTH`.
            unsafe { u32::unchecked_pack(BIT_WIDTH, &encoded, out) };
            references.push(reference);
        }
        Self {
            packed: packed.freeze(),
            references: references.freeze(),
            len: values.len(),
        }
    }
}

/// Unpacks each chunk into the program's scratch.
struct UnpackSource {
    packed: Buffer<u32>,
    validity: Mask,
}

impl BlockSource for UnpackSource {
    fn validity(&self) -> &Mask {
        &self.validity
    }

    fn fill<'s>(
        &'s mut self,
        rows: Range<usize>,
        scratch: &mut BlockBuf,
        _validity: &mut BlockValidity<'_>,
    ) -> VortexResult<SourceBlock<'s>> {
        let chunk = rows.start / CHUNK;
        let packed = &self.packed[chunk * PACKED..][..PACKED];
        // SAFETY: `packed` is one chunk at `BIT_WIDTH` and the scratch holds a chunk.
        unsafe { u32::unchecked_unpack(BIT_WIDTH, packed, scratch.values_mut::<u32>(CHUNK)) };
        Ok(SourceBlock::Scratch {
            offset: 0,
            len: rows.len(),
        })
    }
}

/// Adds each block's chunk reference.
struct AddReference {
    references: Buffer<u32>,
}

impl BlockOp for AddReference {
    fn process<'a>(&mut self, block: Block<'a>) -> VortexResult<Written<'a>> {
        let reference = self.references[block.start() / CHUNK];
        Ok(block.map(|v: u32| v.wrapping_add(reference)))
    }
}

fn program(packed: &Packed, ops: bool) -> BlockProgram {
    let source = UnpackSource {
        packed: packed.packed.clone(),
        validity: Mask::new_true(packed.len),
    };
    let mut program = BlockProgram::new(Box::new(source), PType::U32, packed.len);
    if ops {
        let op = AddReference {
            references: packed.references.clone(),
        };
        program.push(Box::new(op), PType::U32);
    }
    program
}

/// Unpacks every chunk straight into a new buffer, prefetching it as the fused decoders do, then
/// adds its reference there.
fn direct(packed: &Packed, add_reference: bool) -> Buffer<u32> {
    let mut output = BufferMut::<u32>::with_capacity(packed.len);
    // SAFETY: `u32` has no invalid bit patterns, and every chunk is unpacked before it is read.
    unsafe { output.set_len(packed.len) };
    for ((out, chunk), reference) in output
        .chunks_exact_mut(CHUNK)
        .zip(packed.packed.chunks_exact(PACKED))
        .zip(packed.references.iter())
    {
        prefetch_write(out);
        // SAFETY: `chunk` is one chunk at `BIT_WIDTH` and `out` holds a chunk.
        unsafe { u32::unchecked_unpack(BIT_WIDTH, chunk, out) };
        if add_reference {
            out.iter_mut().for_each(|v| *v = v.wrapping_add(*reference));
        }
    }
    output.freeze()
}

/// The same values as a FoR array over a BitPacked child, and the BitPacked child alone.
fn arrays(values: Buffer<u32>) -> (ArrayRef, ArrayRef) {
    let mut ctx = SESSION.create_execution_ctx();
    let array = PrimitiveArray::new(values, Validity::NonNullable);
    let for_array = FoR::encode_chunked(array, &mut ctx).unwrap();
    let FoRReferences::Blocked(references) = for_array.references() else {
        vortex_panic!("every chunk has its own reference");
    };
    let bitpacked = BitPacked::encode(for_array.encoded(), BIT_WIDTH as u8, &mut ctx)
        .unwrap()
        .into_array();
    let for_array = FoR::try_new_chunked(bitpacked.clone(), references.clone(), 0)
        .unwrap()
        .into_array();
    mark_optimized(&for_array, &SESSION).unwrap();
    mark_optimized(&bitpacked, &SESSION).unwrap();
    (for_array, bitpacked)
}

fn bench_program(bencher: Bencher, bytes: usize, ops: bool) {
    let packed = Packed::new(&values(bytes / 4));
    bencher
        .counter(BytesCount::new(bytes))
        .with_inputs(|| (program(&packed, ops), SESSION.create_execution_ctx()))
        .bench_values(|(program, mut ctx)| {
            program.run(Nullability::NonNullable, &mut ctx).unwrap()
        });
}

fn bench_fused(bencher: Bencher, bytes: usize, for_child: bool) {
    let (for_array, bitpacked) = arrays(values(bytes / 4));
    let array = if for_child { for_array } else { bitpacked };
    bencher
        .counter(BytesCount::new(bytes))
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| (*array).clone().execute::<PrimitiveArray>(ctx).unwrap());
}

/// A BitPacked source and a FoR op, run block by block.
#[divan::bench(args = OUTPUT_BYTES)]
fn program_for(bencher: Bencher, bytes: usize) {
    bench_program(bencher, bytes, true);
}

/// Unpacking into the output, then adding the references there.
#[divan::bench(args = OUTPUT_BYTES)]
fn direct_for(bencher: Bencher, bytes: usize) {
    let packed = Packed::new(&values(bytes / 4));
    bencher
        .counter(BytesCount::new(bytes))
        .bench(|| direct(&packed, true));
}

/// The FoR array's own decode, which unpacks and adds in one pass.
#[divan::bench(args = OUTPUT_BYTES)]
fn fused_for(bencher: Bencher, bytes: usize) {
    bench_fused(bencher, bytes, true);
}

/// A BitPacked source alone, whose blocks the program copies into the output.
#[divan::bench(args = OUTPUT_BYTES)]
fn program_bitpacked(bencher: Bencher, bytes: usize) {
    bench_program(bencher, bytes, false);
}

/// Unpacking straight into the output.
#[divan::bench(args = OUTPUT_BYTES)]
fn direct_bitpacked(bencher: Bencher, bytes: usize) {
    let packed = Packed::new(&values(bytes / 4));
    bencher
        .counter(BytesCount::new(bytes))
        .bench(|| direct(&packed, false));
}

/// The BitPacked array's own decode.
#[divan::bench(args = OUTPUT_BYTES)]
fn fused_bitpacked(bencher: Bencher, bytes: usize) {
    bench_fused(bencher, bytes, false);
}
