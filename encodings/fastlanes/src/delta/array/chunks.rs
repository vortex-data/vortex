// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Random-access decoding of the chunks of a Delta array.

use std::cell::RefCell;
use std::mem::MaybeUninit;
use std::ops::Range;

use fastlanes::BitPacking;
use fastlanes::Delta;
use fearless_simd::Level;
use itertools::Itertools;
use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::filter::ChunkDecoder;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::BitWidthsView;
use crate::FL_CHUNK_SIZE;
use crate::bitpack_decompress::execute_block_offsets;
use crate::delta::array::DeltaSlots;
use crate::delta::array::delta_decompress::PackedDeltas;
use crate::delta::array::transpose::TransposeKernel;

/// Decodes any chunk of a Delta array into its 1024 values in logical order.
///
/// Building one executes the bases and the deltas, or for bit-packed deltas locates their packed
/// blocks and patches, so that decoding a chunk afterwards cannot fail.
pub(crate) struct DeltaChunks<'a, T, const LANES: usize> {
    bases: &'a [T],
    deltas: Deltas<'a, T>,
    num_chunks: usize,
    untranspose: TransposeKernel<T>,
    /// Reused buffers for one chunk. A fresh stack buffer per chunk made decoding about 1.7x
    /// slower.
    scratch: RefCell<Box<Scratch<T>>>,
}

/// A transposed chunk, and a chunk in logical order for picking values out of.
struct Scratch<T> {
    transposed: [T; FL_CHUNK_SIZE],
    logical: [MaybeUninit<T>; FL_CHUNK_SIZE],
}

enum Deltas<'a, T> {
    Primitive(&'a [T]),
    Packed(PackedChunks<'a, T>),
}

impl<'a, T, const LANES: usize> DeltaChunks<'a, T, LANES>
where
    T: NativePType + Delta,
{
    /// Decode chunks of `array`, whose bases are `bases` reinterpreted as unsigned.
    ///
    /// Bit-packed deltas are decoded in place. Other deltas are executed into `deltas`, which
    /// the decoder borrows.
    pub(crate) fn for_array(
        array: ArrayView<'a, crate::Delta>,
        bases: &'a [T],
        deltas: &'a mut Option<PrimitiveArray>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let delta_child: &'a ArrayRef = array.slots()[DeltaSlots::DELTAS]
            .as_ref()
            .vortex_expect("Delta deltas slot");
        if let Some(packed) = PackedDeltas::try_new(delta_child) {
            return Self::packed(bases, packed, delta_child.len(), ctx);
        }
        let executed = delta_child
            .clone()
            .execute::<PrimitiveArray>(ctx)?
            .reinterpret_cast(T::PTYPE);
        Self::primitive(bases, deltas.insert(executed).as_slice())
    }

    /// Decode chunks from `bases` and primitive `deltas`, which hold whole chunks.
    pub(crate) fn primitive(bases: &'a [T], deltas: &'a [T]) -> VortexResult<Self> {
        vortex_ensure!(
            deltas.len().is_multiple_of(FL_CHUNK_SIZE),
            "deltas must be padded to a multiple of {FL_CHUNK_SIZE}"
        );
        Self::new(
            bases,
            Deltas::Primitive(deltas),
            deltas.len() / FL_CHUNK_SIZE,
        )
    }

    /// Decode chunks from `bases` and the bit-packed `deltas`, which hold `len` values.
    pub(crate) fn packed(
        bases: &'a [T],
        deltas: PackedDeltas<'a>,
        len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            len.is_multiple_of(FL_CHUNK_SIZE),
            "deltas must be padded to a multiple of {FL_CHUNK_SIZE}"
        );
        let num_chunks = len / FL_CHUNK_SIZE;
        let packed = PackedChunks::new(deltas, num_chunks, ctx)?;
        Self::new(bases, Deltas::Packed(packed), num_chunks)
    }

    fn new(bases: &'a [T], deltas: Deltas<'a, T>, num_chunks: usize) -> VortexResult<Self> {
        vortex_ensure!(
            bases.len() == num_chunks * LANES,
            "expected {} bases for {num_chunks} chunks, got {}",
            num_chunks * LANES,
            bases.len()
        );
        Ok(Self {
            bases,
            deltas,
            num_chunks,
            untranspose: TransposeKernel::untranspose(Level::new()),
            scratch: RefCell::new(Box::new(Scratch {
                transposed: [T::default(); FL_CHUNK_SIZE],
                logical: [const { MaybeUninit::uninit() }; FL_CHUNK_SIZE],
            })),
        })
    }

    /// Decode chunk `chunk` into `output`, in the transposed order of the deltas.
    fn decode_transposed(&self, chunk: usize, output: &mut [T; FL_CHUNK_SIZE]) {
        let base = self.bases[chunk * LANES..]
            .first_chunk::<LANES>()
            .vortex_expect("one base per lane");
        match &self.deltas {
            Deltas::Primitive(deltas) => {
                let deltas = deltas[chunk * FL_CHUNK_SIZE..]
                    .first_chunk::<FL_CHUNK_SIZE>()
                    .vortex_expect("deltas hold whole chunks");
                Delta::undelta::<LANES>(deltas, base, output);
            }
            Deltas::Packed(packed) => {
                let (block, bit_width) = packed.block(chunk);
                let patches = packed.patches(chunk);
                if patches.is_empty() {
                    // SAFETY: `PackedChunks::new` checked that every block holds one chunk at its
                    // bit width.
                    unsafe { Delta::unchecked_undelta_pack(bit_width, block, base, output) };
                } else {
                    let mut unpacked = [T::default(); FL_CHUNK_SIZE];
                    // SAFETY: as above.
                    unsafe { BitPacking::unchecked_unpack(bit_width, block, &mut unpacked) };
                    for &(index, value) in patches {
                        unpacked[index] = value;
                    }
                    Delta::undelta::<LANES>(&unpacked, base, output);
                }
            }
        }
    }

    /// Decode every chunk into a buffer of `num_chunks * 1024` values.
    pub(crate) fn decode_all(&self) -> Buffer<T> {
        let len = self.num_chunks * FL_CHUNK_SIZE;
        let mut output = BufferMut::<T>::with_capacity(len);
        let (chunks, _) = output.spare_capacity_mut()[..len].as_chunks_mut::<FL_CHUNK_SIZE>();
        for (chunk, output) in chunks.iter_mut().enumerate() {
            self.decode_chunk(chunk, output);
        }
        // SAFETY: every chunk was initialized above.
        unsafe { output.set_len(len) };
        output.freeze()
    }
}

// SAFETY: `decode_chunk` untransposes into every value of `dst`, and `decode_indices` writes one
// value for each index.
unsafe impl<T, const LANES: usize> ChunkDecoder<T> for DeltaChunks<'_, T, LANES>
where
    T: NativePType + Delta,
{
    fn decode_chunk(&self, chunk_idx: usize, dst: &mut [MaybeUninit<T>; FL_CHUNK_SIZE]) {
        let transposed = &mut self.scratch.borrow_mut().transposed;
        self.decode_transposed(chunk_idx, transposed);
        self.untranspose.apply_uninit(transposed, dst);
    }

    fn decode_indices(&self, chunk_idx: usize, indices: &[usize], dst: &mut [MaybeUninit<T>]) {
        // A value depends on the earlier values of its lane, so the whole chunk is decoded.
        let mut scratch = self.scratch.borrow_mut();
        let Scratch {
            transposed,
            logical,
        } = &mut **scratch;
        self.decode_transposed(chunk_idx, transposed);
        self.untranspose.apply_uninit(transposed, logical);
        for (dst, &index) in dst.iter_mut().zip_eq(indices) {
            // SAFETY: the untranspose initialized every value of the chunk.
            dst.write(unsafe { logical[index].assume_init() });
        }
    }
}

/// The packed blocks of bit-packed deltas, one per chunk, and their patches by chunk.
struct PackedChunks<'a, T> {
    packed: &'a [T],
    blocks: Blocks,
    /// The patches within the chunks, as (position in chunk, value), grouped by chunk.
    patches: Vec<(usize, T)>,
    /// `patches[patch_starts[c]..patch_starts[c + 1]]` are the patches of chunk `c`.
    patch_starts: Vec<usize>,
}

/// Where the packed block of each chunk is.
enum Blocks {
    /// Every block has `bit_width` bits per value; chunk `c` is block `first + c`.
    Global { bit_width: usize, first: usize },
    /// Block `b` lies between byte boundaries `offsets[b]` and `offsets[b + 1]`, relative to
    /// `offsets[0]`; chunk `c` is block `first + c`.
    Blocked { offsets: Buffer<u64>, first: usize },
}

impl<'a, T: NativePType + BitPacking> PackedChunks<'a, T> {
    fn new(
        deltas: PackedDeltas<'a>,
        num_chunks: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let bp = deltas.packed();
        let first = deltas.first_block();
        let packed = bp.data().packed_slice::<T>();
        let blocks = match bp.bit_widths() {
            BitWidthsView::Global(bit_width) => {
                let bit_width = usize::from(bit_width);
                let words = FL_CHUNK_SIZE * bit_width / (8 * size_of::<T>());
                vortex_ensure!(
                    (first + num_chunks) * words <= packed.len(),
                    "packed deltas hold too few blocks"
                );
                Blocks::Global { bit_width, first }
            }
            BitWidthsView::Blocked(offsets) => {
                // The validated offsets bound every block to its bit width within `packed`.
                let offsets = execute_block_offsets(bp, offsets, ctx)?;
                vortex_ensure!(
                    first + num_chunks < offsets.len(),
                    "packed deltas hold too few blocks"
                );
                Blocks::Blocked { offsets, first }
            }
        };
        let blocks_range = first..first + num_chunks;
        let patches = chunk_patches::<T>(bp, &blocks_range, ctx)?;
        let mut patch_starts = Vec::with_capacity(num_chunks + 1);
        let mut start = 0;
        for chunk in 0..=num_chunks {
            start +=
                patches[start..].partition_point(|&(position, _)| position < chunk * FL_CHUNK_SIZE);
            patch_starts.push(start);
        }
        let patches = patches
            .into_iter()
            .map(|(position, value)| (position % FL_CHUNK_SIZE, value))
            .collect();
        Ok(Self {
            packed,
            blocks,
            patches,
            patch_starts,
        })
    }

    /// The packed block of chunk `chunk` and its bit width.
    fn block(&self, chunk: usize) -> (&'a [T], usize) {
        match &self.blocks {
            Blocks::Global { bit_width, first } => {
                let words = FL_CHUNK_SIZE * bit_width / (8 * size_of::<T>());
                (&self.packed[(first + chunk) * words..][..words], *bit_width)
            }
            Blocks::Blocked { offsets, first } => {
                let block = first + chunk;
                let base = offsets[0];
                // Validation bounds these differences by the packed buffer's usize length.
                let start = (offsets[block] - base) as usize;
                let end = (offsets[block + 1] - base) as usize;
                (
                    &self.packed[start / size_of::<T>()..end / size_of::<T>()],
                    (end - start) / 128,
                )
            }
        }
    }

    /// The patches of chunk `chunk`, as (position in chunk, value).
    fn patches(&self, chunk: usize) -> &[(usize, T)] {
        &self.patches[self.patch_starts[chunk]..self.patch_starts[chunk + 1]]
    }
}

/// The patches of `bp` that fall within `blocks`, as positions counted from the start of the
/// first block, in ascending order.
fn chunk_patches<T: NativePType>(
    bp: ArrayView<'_, BitPacked>,
    blocks: &Range<usize>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<(usize, T)>> {
    let Some(patches) = bp.patches() else {
        return Ok(Vec::new());
    };
    let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
    let values = patches
        .values()
        .clone()
        .execute::<PrimitiveArray>(ctx)?
        .reinterpret_cast(T::PTYPE);
    let values = values.as_slice::<T>();
    let start = blocks.start * FL_CHUNK_SIZE;
    let positions = start..blocks.end * FL_CHUNK_SIZE;
    // Patch indices count from the start of `bp`, which sits `offset` into its first block.
    let shift = usize::from(bp.offset());
    let mut chunk_patches: Vec<(usize, T)> =
        match_each_unsigned_integer_ptype!(indices.ptype(), |P| {
            indices
                .as_slice::<P>()
                .iter()
                .zip_eq(values)
                .filter_map(|(&index, &value)| {
                    let position = AsPrimitive::<usize>::as_(index) - patches.offset() + shift;
                    positions
                        .contains(&position)
                        .then(|| (position - start, value))
                })
                .collect()
        });
    chunk_patches.sort_unstable_by_key(|&(position, _)| position);
    Ok(chunk_patches)
}
