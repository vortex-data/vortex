// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter::Chain;
use std::iter::Once;
use std::iter::once;
use std::sync::Arc;

use vortex_buffer::BitBuffer;
use vortex_buffer::BitChunkIterator;
use vortex_buffer::BufferMut;
use vortex_buffer::CpuKernel;
use vortex_error::VortexExpect;

use crate::Mask;
use crate::MaskValues;
use crate::MaskValuesRef;

#[cfg(target_arch = "x86_64")]
mod bmi2;
#[cfg(target_arch = "aarch64")]
mod sve;

trait DepositBits {
    fn deposit_bits(source: u64, mask: u64, mask_count: usize) -> u64;
}

trait SelectBit {
    /// Position (0..63) of the `rank`-th set bit in `word`. Caller ensures
    /// `rank < word.count_ones()`.
    fn select_bit_position(word: u64, rank: usize) -> usize;
}

/// Portable deposit without branches on the data: closed forms for masks with at most two set
/// or two clear bits, a byte-wise network otherwise.
struct Portable;

impl DepositBits for Portable {
    // Inlined into the kernel loop: as a call per word it costs more than the closed forms.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn deposit_bits(source: u64, mask: u64, mask_count: usize) -> u64 {
        match mask_count {
            0..=2 => deposit_two_lowest(source, mask),
            62.. => deposit_two_holes(source, mask),
            _ => deposit_bytewise(source, mask),
        }
    }
}

impl SelectBit for Portable {
    #[inline]
    fn select_bit_position(word: u64, rank: usize) -> usize {
        select_bit_position_portable(word, rank)
    }
}

/// Portable deposit for a sparse `self`: a loop over the mask bits that stops once the source
/// runs out. With a few bits per chunk it beats the fixed cost of [`Portable`].
struct PortableLoop;

impl DepositBits for PortableLoop {
    #[inline]
    fn deposit_bits(mut source: u64, mut mask: u64, mask_count: usize) -> u64 {
        if mask_count <= 2 {
            return deposit_two_lowest(source, mask);
        }

        let mut result = 0u64;
        while mask != 0 && source != 0 {
            let bit = mask & mask.wrapping_neg();
            result |= bit & (source & 1).wrapping_neg();
            source >>= 1;
            mask &= mask - 1;
        }
        result
    }
}

/// `b` in every byte of a word.
const fn bytes(b: u8) -> u64 {
    b as u64 * 0x0101_0101_0101_0101
}

/// Deposit into the lowest two set bits of `mask`.
#[allow(clippy::inline_always)]
#[inline(always)]
fn deposit_two_lowest(source: u64, mask: u64) -> u64 {
    let lowest = mask & mask.wrapping_neg();
    let rest = mask ^ lowest;
    let second = rest & rest.wrapping_neg();
    (lowest & (source & 1).wrapping_neg()) | (second & ((source >> 1) & 1).wrapping_neg())
}

/// Deposit into a mask with at most two clear bits: the source moves up by one past each hole.
#[allow(clippy::inline_always)]
#[inline(always)]
fn deposit_two_holes(mut source: u64, mask: u64) -> u64 {
    let mut holes = !mask;
    for _ in 0..2 {
        let hole = holes & holes.wrapping_neg();
        let below = hole.wrapping_sub(1);
        source = (source & below) | ((source << 1) & !below);
        holes ^= hole;
    }
    source & mask
}

/// Deposit a byte at a time.
///
/// Byte `j` takes its source bits from the number of mask bits in the bytes below it, which
/// one multiply sums for all bytes. Within each byte the bits are spread by the compress network
/// of Hacker's Delight 7-4 run backwards, as in `uN::extract_bits` of the Rust core library:
/// three shift-and-XOR stages for 8 bits instead of six for 64.
#[allow(clippy::inline_always)]
#[inline(always)]
fn deposit_bytewise(source: u64, mask: u64) -> u64 {
    let pairs = mask - ((mask >> 1) & bytes(0x55));
    let nibbles = (pairs & bytes(0x33)) + ((pairs >> 2) & bytes(0x33));
    let counts = (nibbles + (nibbles >> 4)) & bytes(0x0F);
    // Byte j holds the mask bits in bytes 0..j. The sums stay below 64, so no byte carries.
    let below = counts.wrapping_mul(bytes(1)) << 8;

    // Compress the mask within each byte. Stage i moves down by 2^i the bits whose count of mask
    // zeros below them has bit i set; `landed` keeps where they land. Before stage i the zeros
    // stand in aligned groups of 2^i, so the prefix XOR can start at a shift of 2^i.
    let mut packed = mask;
    let mut zeros = !mask;
    let mut landed = [0u64; 3];
    for (stage, landed) in landed.iter_mut().enumerate() {
        let shift = 1 << stage;
        let mut parity = zeros;
        let mut len = shift;
        while len < 8 {
            parity ^= (parity << len) & bytes(0xFF << len);
            len <<= 1;
        }
        let moving = packed & parity;
        *landed = (moving >> shift) & bytes(0xFF >> shift);
        packed ^= moving ^ *landed;
        zeros &= !parity;
        zeros ^= (zeros >> shift) & bytes(0xFF >> shift);
    }

    // Pack the source bits of each byte at its low end, then undo the stages.
    let mut result = 0;
    for byte in 0..8 {
        let from = (below >> (8 * byte)) & 0xFF;
        result |= ((source >> from) & 0xFF) << (8 * byte);
    }
    result &= packed;
    for (stage, landed) in landed.iter().enumerate().rev() {
        result = (result & !landed) | ((result & landed) << (1 << stage));
    }
    result & mask
}

#[inline]
fn select_bit_position_portable(word: u64, mut rank: usize) -> usize {
    debug_assert!(rank < word.count_ones() as usize);
    let mut bit_offset = 0usize;
    for byte in word.to_le_bytes() {
        let count = byte.count_ones() as usize;
        if rank < count {
            let mut bits = byte;
            for _ in 0..rank {
                bits &= bits - 1;
            }

            return bit_offset + bits.trailing_zeros() as usize;
        }

        rank -= count;
        bit_offset += 8;
    }

    debug_assert!(false, "rank out of bounds");
    0
}

/// Reader that pulls variable-length (0..=64 bit) groups from a [`BitBuffer`] sequentially.
///
/// Maintains a 128-bit window over two consecutive chunks (`current`, `next`) and uses a
/// funnel shift via `u128` to extract bits at any offset without branching. The shift
/// pattern compiles to a single funnel-shift / SHRD-style sequence on x86_64.
struct RankBitReader<'a> {
    chunk_iter: Chain<BitChunkIterator<'a>, Once<u64>>,
    current: u64,
    next: u64,
    bit_offset: usize,
}

impl<'a> RankBitReader<'a> {
    fn new(buffer: &'a BitBuffer) -> Self {
        let chunks = buffer.chunks();
        let mut chunk_iter = chunks.iter().chain(once(chunks.remainder_bits()));

        let current = chunk_iter.next().unwrap_or(0);
        let next = chunk_iter.next().unwrap_or(0);

        Self {
            chunk_iter,
            current,
            next,
            bit_offset: 0,
        }
    }

    #[inline]
    fn fetch_next(&mut self) -> u64 {
        self.chunk_iter.next().unwrap_or(0)
    }

    #[inline]
    fn read(&mut self, bit_count: usize) -> u64 {
        debug_assert!(bit_count <= 64);

        // Funnel shift: extract `bit_count` bits at `bit_offset` from the (next:current)
        // 128-bit window. For bit_offset in 0..=63 this is a single SHRD-style instruction
        // on x86_64; the u128 cast keeps it well-defined when bit_offset == 0.
        let combined = ((self.next as u128) << 64) | (self.current as u128);
        // The truncation is intentional: we want the low 64 bits of the funnel-shifted
        // window, which is exactly what `as u64` produces.
        #[expect(clippy::cast_possible_truncation)]
        let bits = (combined >> self.bit_offset) as u64 & low_bits(bit_count);

        let new_offset = self.bit_offset + bit_count;
        if new_offset >= 64 {
            self.current = self.next;
            self.next = self.fetch_next();
            self.bit_offset = new_offset - 64;
        } else {
            self.bit_offset = new_offset;
        }

        bits
    }
}

#[inline]
fn low_bits(bit_count: usize) -> u64 {
    debug_assert!(bit_count <= 64);
    if bit_count == 64 {
        u64::MAX
    } else {
        (1u64 << bit_count) - 1
    }
}

#[inline]
fn mask_from_buffer(buffer: BitBuffer, true_count: usize) -> Mask {
    let len = buffer.len();
    if true_count == 0 {
        return Mask::new_false(len);
    }
    if true_count == len {
        return Mask::new_true(len);
    }

    Mask::Values(Arc::new(MaskValues {
        buffer,
        indices: Default::default(),
        slices: Default::default(),
        true_count,
        density: true_count as f64 / len as f64,
    }))
}

#[inline]
fn push_result_chunk<D: DepositBits>(
    result: &mut BufferMut<u64>,
    self_chunk: u64,
    self_count: usize,
    rank_bits: u64,
) {
    let chunk = D::deposit_bits(rank_bits, self_chunk, self_count);
    // SAFETY: callers allocate enough capacity for every output chunk.
    unsafe { result.push_unchecked(chunk) };
}

/// Marked `#[inline(always)]` so each `#[target_feature]` wrapper in `bmi2` and `sve` gets its own
/// fully-inlined copy compiled with that feature set, as in `vortex_buffer::bit::pack`.
#[allow(clippy::inline_always)]
#[inline(always)]
fn intersect_bit_buffers<D: DepositBits>(
    self_buffer: &BitBuffer,
    mask_buffer: &BitBuffer,
    true_count: usize,
) -> Mask {
    let len = self_buffer.len();
    let mut result = BufferMut::with_capacity(len.div_ceil(64));
    let mut reader = RankBitReader::new(mask_buffer);
    let self_chunks = self_buffer.chunks();

    for self_chunk in self_chunks.iter() {
        let self_count = self_chunk.count_ones() as usize;
        let rank_bits = reader.read(self_count);
        push_result_chunk::<D>(&mut result, self_chunk, self_count, rank_bits);
    }

    if self_chunks.remainder_len() != 0 {
        let self_chunk = self_chunks.remainder_bits();
        let self_count = self_chunk.count_ones() as usize;
        let rank_bits = reader.read(self_count);
        push_result_chunk::<D>(&mut result, self_chunk, self_count, rank_bits);
    }

    mask_from_buffer(
        BitBuffer::new(result.freeze().into_byte_buffer(), len),
        true_count,
    )
}

#[allow(clippy::inline_always)]
#[inline(always)]
fn intersect_bit_buffer_by_rank_indices<D: DepositBits>(
    self_buffer: &BitBuffer,
    mask_indices: &[usize],
) -> Mask {
    let len = self_buffer.len();
    let mut result = BufferMut::with_capacity(len.div_ceil(64));
    let self_chunks = self_buffer.chunks();
    let mut rank_base = 0usize;
    let mut rank_idx = 0usize;

    for self_chunk in self_chunks.iter() {
        let self_count = self_chunk.count_ones() as usize;
        let next_rank_base = rank_base + self_count;
        let rank_bits = rank_bits_for_chunk(mask_indices, &mut rank_idx, rank_base, next_rank_base);
        push_result_chunk::<D>(&mut result, self_chunk, self_count, rank_bits);
        rank_base = next_rank_base;
    }

    if self_chunks.remainder_len() != 0 {
        let self_chunk = self_chunks.remainder_bits();
        let self_count = self_chunk.count_ones() as usize;
        let next_rank_base = rank_base + self_count;
        let rank_bits = rank_bits_for_chunk(mask_indices, &mut rank_idx, rank_base, next_rank_base);
        push_result_chunk::<D>(&mut result, self_chunk, self_count, rank_bits);
    }

    debug_assert_eq!(rank_idx, mask_indices.len());

    mask_from_buffer(
        BitBuffer::new(result.freeze().into_byte_buffer(), len),
        mask_indices.len(),
    )
}

/// Walks `mask_indices` (global ranks into `self_buffer.set_bits`) and emits the corresponding
/// positions in `self_buffer`. For each rank, advances `self_buffer`'s chunks via popcount
/// skip-while, then locates the bit inside the current chunk with rank-select.
///
/// This dominates the chunk-scan paths when the mask is very sparse: cost is
/// `O(mask.true_count() + self.len() / 64)` rather than `O(self.len() / 64)` per chunk.
#[allow(clippy::inline_always)]
#[inline(always)]
fn intersect_mask_driven<S, I>(self_buffer: &BitBuffer, mask_indices: I, true_count: usize) -> Mask
where
    S: SelectBit,
    I: Iterator<Item = usize>,
{
    let len = self_buffer.len();
    if true_count == 0 {
        return Mask::new_false(len);
    }

    let mut chunk_iter = self_buffer.chunks().iter_padded();

    let mut current_chunk = chunk_iter.next().unwrap_or(0);
    let mut current_count = current_chunk.count_ones() as usize;
    let mut current_chunk_idx = 0usize;
    let mut rank_before = 0usize;

    let mut output = Vec::with_capacity(true_count);

    for global_rank in mask_indices {
        while rank_before + current_count <= global_rank {
            rank_before += current_count;
            current_chunk_idx += 1;
            current_chunk = chunk_iter.next().vortex_expect("mask index out of bounds");
            current_count = current_chunk.count_ones() as usize;
        }

        let local_rank = global_rank - rank_before;
        let bit_pos = S::select_bit_position(current_chunk, local_rank);
        output.push(current_chunk_idx * 64 + bit_pos);
    }

    debug_assert_eq!(output.len(), true_count);
    Mask::from_indices(len, output)
}

#[inline]
fn rank_bits_for_chunk(
    mask_indices: &[usize],
    rank_idx: &mut usize,
    rank_base: usize,
    next_rank_base: usize,
) -> u64 {
    let mut rank_bits = 0u64;
    while let Some(&rank) = mask_indices.get(*rank_idx) {
        if rank >= next_rank_base {
            break;
        }
        rank_bits |= 1u64 << (rank - rank_base);
        *rank_idx += 1;
    }
    rank_bits
}

fn intersect_by_rank_indices(len: usize, self_indices: &[usize], mask_indices: &[usize]) -> Mask {
    Mask::from_indices(
        len,
        mask_indices.iter().map(|idx| {
            // SAFETY: mask indices are ranks into self_indices, because
            // mask.len() == self.true_count() == self_indices.len().
            unsafe { *self_indices.get_unchecked(*idx) }
        }),
    )
}

/// Below this density of `self` the portable path deposits with [`PortableLoop`]. See the
/// `random_rotating` benchmark when changing it.
const PORTABLE_LOOP_MAX_DENSITY: f64 = 1.0 / 16.0;

type IntersectBuffers = unsafe fn(&BitBuffer, &BitBuffer, usize) -> Mask;
type IntersectRankIndices = unsafe fn(&BitBuffer, &[usize]) -> Mask;

/// The bit-buffer kernel for this CPU: its deposit instruction if it has one, otherwise `P`.
fn select_bit_buffers<P: DepositBits>() -> IntersectBuffers {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("bmi2") {
            return bmi2::intersect_bit_buffers_bmi2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("sve2-bitperm") {
            return sve::intersect_bit_buffers_sve2;
        }
    }
    intersect_bit_buffers::<P>
}

/// The rank-indices kernel for this CPU, as in [`select_bit_buffers`].
fn select_rank_indices<P: DepositBits>() -> IntersectRankIndices {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("bmi2") {
            return bmi2::intersect_bit_buffer_by_rank_indices_bmi2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("sve2-bitperm") {
            return sve::intersect_bit_buffer_by_rank_indices_sve2;
        }
    }
    intersect_bit_buffer_by_rank_indices::<P>
}

#[inline]
fn intersect_bit_buffers_dispatch(
    self_buffer: &BitBuffer,
    mask_buffer: &BitBuffer,
    true_count: usize,
    self_density: f64,
) -> Mask {
    static KERNEL: CpuKernel<IntersectBuffers> = CpuKernel::new(select_bit_buffers::<Portable>);
    static SPARSE_KERNEL: CpuKernel<IntersectBuffers> =
        CpuKernel::new(select_bit_buffers::<PortableLoop>);
    let kernel = if self_density < PORTABLE_LOOP_MAX_DENSITY {
        &SPARSE_KERNEL
    } else {
        &KERNEL
    };
    // SAFETY: the selector only returns kernels that are safe or whose required CPU
    // features were probed before selection.
    unsafe { kernel.get()(self_buffer, mask_buffer, true_count) }
}

#[inline]
fn intersect_rank_indices_dispatch(
    self_buffer: &BitBuffer,
    mask_indices: &[usize],
    self_density: f64,
) -> Mask {
    static KERNEL: CpuKernel<IntersectRankIndices> =
        CpuKernel::new(select_rank_indices::<Portable>);
    static SPARSE_KERNEL: CpuKernel<IntersectRankIndices> =
        CpuKernel::new(select_rank_indices::<PortableLoop>);
    let kernel = if self_density < PORTABLE_LOOP_MAX_DENSITY {
        &SPARSE_KERNEL
    } else {
        &KERNEL
    };
    // SAFETY: the selector only returns kernels that are safe or whose required CPU
    // features were probed before selection.
    unsafe { kernel.get()(self_buffer, mask_indices) }
}

#[inline]
fn intersect_mask_driven_dispatch<I>(
    self_buffer: &BitBuffer,
    mask_indices: I,
    true_count: usize,
) -> Mask
where
    I: Iterator<Item = usize>,
{
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("bmi2") {
        // SAFETY: BMI2 was just detected.
        return unsafe { bmi2::intersect_mask_driven_bmi2(self_buffer, mask_indices, true_count) };
    }

    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("sve2-bitperm") {
        // SAFETY: SVE2 BITPERM was just detected.
        return unsafe { sve::intersect_mask_driven_sve2(self_buffer, mask_indices, true_count) };
    }

    intersect_mask_driven::<Portable, _>(self_buffer, mask_indices, true_count)
}

/// Returns whether a mask is sparse.
///
/// [`BitBuffer`] traversal uses `u64` words, so fewer than one selected value per word is sparse.
fn mask_is_sparse(values: &MaskValuesRef) -> bool {
    values.true_count().saturating_mul(64) < values.len()
}

/// Returns whether a rank mask is sparse.
///
/// The mask-driven path becomes worthwhile at approximately 3% mask density. Each set bit costs a
/// select and push, but this avoids a popcount and deposit for each chunk of `self`.
fn rank_mask_is_sparse(values: &MaskValuesRef) -> bool {
    values.true_count().saturating_mul(32) < values.len()
}

impl Mask {
    /// Take the intersection of the `mask` with the set of true values in `self`.
    ///
    /// The hot path keeps bit-buffer-backed masks as bit buffers. It scans the set bits of `self`
    /// by rank and deposits selected rank bits into their original positions.
    ///
    /// # Examples
    ///
    /// Keep the third and fifth set values from mask `m1`:
    /// ```
    /// use vortex_mask::Mask;
    ///
    /// let m1 = Mask::from_iter([true, false, false, true, true, true, false, true]);
    /// let m2 = Mask::from_iter([false, false, true, false, true]);
    /// assert_eq!(
    ///     m1.intersect_by_rank(&m2),
    ///     Mask::from_iter([false, false, false, false, true, false, false, true])
    /// );
    /// ```
    pub fn intersect_by_rank(&self, mask: &Mask) -> Mask {
        assert_eq!(self.true_count(), mask.len());

        match (self, mask) {
            (Self::AllTrue(_), _) => mask.clone(),
            (_, Self::AllTrue(_)) => self.clone(),
            (Self::AllFalse(_), _) | (_, Self::AllFalse(_)) => Self::new_false(self.len()),
            (Self::Values(self_values), Self::Values(mask_values)) => {
                // Four dispatch cases keyed by (self density, mask density):
                //
                //              | mask sparse | mask dense
                // -------------+-------------+------------
                // self sparse  | indices     | indices
                // self dense   | mask-driven | bit-buffer
                if let Some(mask_indices) = mask_values.indices.get() {
                    if let Some(self_indices) = self_values.indices.get()
                        && mask_indices.len() < self.len().div_ceil(64)
                    {
                        return intersect_by_rank_indices(self.len(), self_indices, mask_indices);
                    }

                    let self_is_very_sparse = mask_is_sparse(self_values);
                    let mask_is_very_sparse = rank_mask_is_sparse(mask_values);

                    if self_is_very_sparse {
                        return intersect_by_rank_indices(
                            self.len(),
                            self_values.indices(),
                            mask_indices,
                        );
                    }

                    if mask_is_very_sparse {
                        return intersect_mask_driven_dispatch(
                            self_values.bit_buffer(),
                            mask_indices.iter().copied(),
                            mask_values.true_count(),
                        );
                    }

                    if mask_indices.len().saturating_mul(4) > mask.len() {
                        return intersect_bit_buffers_dispatch(
                            self_values.bit_buffer(),
                            mask_values.bit_buffer(),
                            mask_values.true_count(),
                            self_values.density(),
                        );
                    }

                    return intersect_rank_indices_dispatch(
                        self_values.bit_buffer(),
                        mask_indices,
                        self_values.density(),
                    );
                }

                let self_is_very_sparse = mask_is_sparse(self_values);
                let mask_is_very_sparse = rank_mask_is_sparse(mask_values);

                if self_is_very_sparse {
                    return intersect_by_rank_indices(
                        self.len(),
                        self_values.indices(),
                        mask_values.indices(),
                    );
                }

                if mask_is_very_sparse {
                    return intersect_mask_driven_dispatch(
                        self_values.bit_buffer(),
                        mask_values.bit_buffer().set_indices(),
                        mask_values.true_count(),
                    );
                }

                intersect_bit_buffers_dispatch(
                    self_values.bit_buffer(),
                    mask_values.bit_buffer(),
                    mask_values.true_count(),
                    self_values.density(),
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::BitBuffer;

    use super::DepositBits;
    use super::Portable;
    use super::PortableLoop;
    #[cfg(target_arch = "aarch64")]
    use super::SelectBit;
    use super::intersect_bit_buffer_by_rank_indices;
    use super::intersect_bit_buffers;
    #[cfg(target_arch = "aarch64")]
    use super::sve::Sve2Bdep;
    use crate::Mask;

    #[test]
    fn mask_bitand_all_as_bit_and() {
        let this = Mask::from_buffer(BitBuffer::from_iter(vec![true, true, true, true, true]));
        let mask = Mask::from_buffer(BitBuffer::from_iter(vec![false, true, false, true, true]));
        assert_eq!(
            this.intersect_by_rank(&mask),
            Mask::from_indices(5, vec![1, 3, 4])
        );
    }

    #[test]
    fn mask_bitand_all_true() {
        let this = Mask::from_buffer(BitBuffer::from_iter(vec![false, false, true, true, true]));
        let mask = Mask::from_buffer(BitBuffer::from_iter(vec![true, true, true]));
        assert_eq!(
            this.intersect_by_rank(&mask),
            Mask::from_indices(5, vec![2, 3, 4])
        );
    }

    #[test]
    fn mask_bitand_true() {
        let this = Mask::from_buffer(BitBuffer::from_iter(vec![true, false, false, true, true]));
        let mask = Mask::from_buffer(BitBuffer::from_iter(vec![true, false, true]));
        assert_eq!(
            this.intersect_by_rank(&mask),
            Mask::from_indices(5, vec![0, 4])
        );
    }

    #[test]
    fn mask_bitand_false() {
        let this = Mask::from_buffer(BitBuffer::from_iter(vec![true, false, false, true, true]));
        let mask = Mask::from_buffer(BitBuffer::from_iter(vec![false, false, false]));
        assert_eq!(this.intersect_by_rank(&mask), Mask::from_indices(5, vec![]));
    }

    #[test]
    fn mask_intersect_by_rank_all_false() {
        let this = Mask::AllFalse(10);
        let mask = Mask::AllFalse(0);
        assert_eq!(this.intersect_by_rank(&mask), Mask::AllFalse(10));
    }

    #[rstest]
    #[case::all_true_with_all_true(
        Mask::new_true(5),
        Mask::new_true(5),
        vec![0, 1, 2, 3, 4]
    )]
    #[case::all_true_with_all_false(
        Mask::new_true(5),
        Mask::new_false(5),
        vec![]
    )]
    #[case::all_false_with_any(
        Mask::new_false(10),
        Mask::new_true(0),
        vec![]
    )]
    #[case::indices_with_all_true(
        Mask::from_indices(10, vec![2, 5, 7, 9]),
        Mask::new_true(4),
        vec![2, 5, 7, 9]
    )]
    #[case::indices_with_all_false(
        Mask::from_indices(10, vec![2, 5, 7, 9]),
        Mask::new_false(4),
        vec![]
    )]
    fn test_intersect_by_rank_special_cases(
        #[case] base_mask: Mask,
        #[case] rank_mask: Mask,
        #[case] expected_indices: Vec<usize>,
    ) {
        let result = base_mask.intersect_by_rank(&rank_mask);

        match result.indices() {
            crate::AllOr::All => assert_eq!(expected_indices.len(), result.len()),
            crate::AllOr::None => assert!(expected_indices.is_empty()),
            crate::AllOr::Some(indices) => assert_eq!(indices, &expected_indices[..]),
        }
    }

    #[test]
    fn test_intersect_by_rank_example() {
        // Example from the documentation
        let m1 = Mask::from_iter([true, false, false, true, true, true, false, true]);
        let m2 = Mask::from_iter([false, false, true, false, true]);
        let result = m1.intersect_by_rank(&m2);
        let expected = Mask::from_iter([false, false, false, false, true, false, false, true]);
        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic]
    fn test_intersect_by_rank_wrong_length() {
        let m1 = Mask::from_indices(10, vec![2, 5, 7]); // 3 true values
        let m2 = Mask::new_true(5); // 5 true values - doesn't match
        m1.intersect_by_rank(&m2);
    }

    #[rstest]
    #[case::single_element(
        vec![3],
        vec![true],
        vec![3]
    )]
    #[case::single_element_masked(
        vec![3],
        vec![false],
        vec![]
    )]
    #[case::alternating(
        vec![0, 2, 4, 6, 8],
        vec![true, false, true, false, true],
        vec![0, 4, 8]
    )]
    #[case::consecutive(
        vec![5, 6, 7, 8, 9],
        vec![false, true, true, true, false],
        vec![6, 7, 8]
    )]
    fn test_intersect_by_rank_patterns(
        #[case] base_indices: Vec<usize>,
        #[case] rank_pattern: Vec<bool>,
        #[case] expected_indices: Vec<usize>,
    ) {
        let base = Mask::from_indices(10, base_indices);
        let rank = Mask::from_iter(rank_pattern);
        let result = base.intersect_by_rank(&rank);

        match result.indices() {
            crate::AllOr::Some(indices) => assert_eq!(indices, &expected_indices[..]),
            crate::AllOr::None => assert!(expected_indices.is_empty()),
            _ => panic!("Unexpected result"),
        }
    }

    #[rstest]
    // Larger sizes to push the bench-shaped buffer paths through the unit tests too.
    #[case::dense_len_1024(1024, 31, 0.5, 0.5)]
    // Very-sparse mask exercises the mask-driven dispatch path. Both densities live in
    // the half-open interval where `mask_is_very_sparse` is true.
    #[case::sparse_mask_1pct(1024, 17, 0.5, 0.01)]
    #[case::sparse_mask_2pct(2048, 0, 0.5, 0.02)]
    #[case::very_sparse_mask_with_offsets(513, 5, 0.5, 0.005)]
    fn test_intersect_by_rank_density_matrix(
        #[case] base_len: usize,
        #[case] base_offset: usize,
        #[case] base_density: f64,
        #[case] rank_density: f64,
    ) {
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let base_threshold = (base_density * 1024.0) as usize;
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let rank_threshold = (rank_density * 1024.0) as usize;

        let base_source: Vec<bool> = (0..base_len + base_offset + 16)
            .map(|i| (i * 7 + 13) % 1024 < base_threshold)
            .collect();
        let base_bits = base_source[base_offset..base_offset + base_len].to_vec();
        let base = Mask::from_buffer(
            BitBuffer::from(base_source).slice(base_offset..base_offset + base_len),
        );

        let rank_len = base.true_count();
        let rank_bits: Vec<bool> = (0..rank_len)
            .map(|i| (i * 11 + 7) % 1024 < rank_threshold)
            .collect();
        let rank_from_buffer = Mask::from_buffer(BitBuffer::from(rank_bits.clone()));
        let rank_indices_vec = rank_bits
            .iter()
            .enumerate()
            .filter_map(|(idx, &v)| v.then_some(idx))
            .collect::<Vec<_>>();
        let rank_from_indices = Mask::from_indices(rank_len, rank_indices_vec);

        let expected = expected_intersect_by_rank(&base_bits, &rank_bits);

        assert_eq!(
            base.intersect_by_rank(&rank_from_buffer),
            expected,
            "uncached rank"
        );
        assert_eq!(
            base.intersect_by_rank(&rank_from_indices),
            expected,
            "cached rank"
        );
    }

    #[rstest]
    #[case::short(37, 0, 0)]
    #[case::base_offset(257, 5, 0)]
    #[case::rank_offset(257, 0, 3)]
    #[case::both_offsets(513, 6, 5)]
    fn test_intersect_by_rank_bitbuffer_paths_with_offsets(
        #[case] base_len: usize,
        #[case] base_offset: usize,
        #[case] rank_offset: usize,
    ) {
        let base_source: Vec<bool> = (0..base_len + base_offset + 16)
            .map(|i| (i % 3 == 0) ^ (i % 11 == 0) ^ (i % 17 == 0))
            .collect();
        let base_bits = base_source[base_offset..base_offset + base_len].to_vec();
        let base = Mask::from_buffer(
            BitBuffer::from(base_source).slice(base_offset..base_offset + base_len),
        );

        let rank_len = base.true_count();
        let rank_bits: Vec<bool> = (0..rank_len)
            .map(|i| (i % 5 == 0) || (i % 13 == 3))
            .collect();
        let mut rank_source = vec![false; rank_offset];
        rank_source.extend(rank_bits.iter().copied());
        rank_source.extend([true, false, true, false, true, false, true, false]);

        let rank_from_buffer = Mask::from_buffer(
            BitBuffer::from(rank_source).slice(rank_offset..rank_offset + rank_len),
        );
        let rank_indices = rank_bits
            .iter()
            .enumerate()
            .filter_map(|(idx, &value)| value.then_some(idx))
            .collect::<Vec<_>>();
        let rank_from_indices = Mask::from_indices(rank_len, rank_indices);

        let expected = expected_intersect_by_rank(&base_bits, &rank_bits);

        assert_eq!(base.intersect_by_rank(&rank_from_buffer), expected);
        assert_eq!(base.intersect_by_rank(&rank_from_indices), expected);
    }

    /// xorshift64, so the portable tests need no dependency and see the same words on every run.
    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn deposit_reference(mut source: u64, mut mask: u64) -> u64 {
        let mut result = 0u64;
        while mask != 0 {
            let bit = mask & mask.wrapping_neg();
            if source & 1 != 0 {
                result |= bit;
            }
            source >>= 1;
            mask &= mask - 1;
        }
        result
    }

    #[test]
    fn portable_deposit_matches_reference() {
        deposit_matches_reference::<Portable>();
        deposit_matches_reference::<PortableLoop>();
    }

    fn deposit_matches_reference<D: DepositBits>() {
        let mut state = 0x9E37_79B9_7F4A_7C15;
        for round in 0..200_000 {
            // Up to six ANDs of random words, inverted every other round: all densities,
            // down to masks with none or all of the bits.
            let mut mask = xorshift(&mut state);
            for _ in 0..round % 7 {
                mask &= xorshift(&mut state);
            }
            if round % 2 == 1 {
                mask = !mask;
            }
            let source = xorshift(&mut state);
            let count = mask.count_ones() as usize;
            assert_eq!(
                D::deposit_bits(source, mask, count),
                deposit_reference(source, mask),
                "source {source:#018x}, mask {mask:#018x}"
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn sve2_bdep_matches_reference() {
        if !std::arch::is_aarch64_feature_detected!("sve2-bitperm") {
            return;
        }
        deposit_matches_reference::<Sve2Bdep>();
        for offset in [0, 3] {
            kernels_match_reference::<Sve2Bdep>(offset);
        }
        let mut state = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..10_000 {
            let word = xorshift(&mut state) | 1;
            for rank in 0..word.count_ones() as usize {
                let bit = deposit_reference(1u64 << rank, word).trailing_zeros() as usize;
                assert_eq!(Sve2Bdep::select_bit_position(word, rank), bit);
            }
        }
    }

    /// The portable kernels directly, as dispatch picks BMI2 or SVE2 where it is available.
    #[rstest]
    #[case::aligned(0)]
    #[case::offset(3)]
    fn portable_kernels_match_reference(#[case] offset: usize) {
        kernels_match_reference::<Portable>(offset);
        kernels_match_reference::<PortableLoop>(offset);
    }

    /// Both bit-buffer kernels with deposit `D`. Runs of empty and full chunks sit between
    /// random ones.
    fn kernels_match_reference<D: DepositBits>(offset: usize) {
        let mut state = 0x2545_F491_4F6C_DD1D;
        let mut base_source = vec![false; offset];
        for run in 0..48 {
            let words = 1 + (xorshift(&mut state) % 20) as usize;
            for _ in 0..words {
                let word = match run % 3 {
                    0 => 0,
                    1 => u64::MAX,
                    _ => xorshift(&mut state),
                };
                base_source.extend((0..64).map(|i| (word >> i) & 1 == 1));
            }
        }
        base_source.extend([true, false, true, true, false]);
        let base_len = base_source.len() - offset;
        let base_bits = base_source[offset..].to_vec();
        let base_buffer = BitBuffer::from(base_source).slice(offset..offset + base_len);

        let rank_len = base_bits.iter().filter(|&&b| b).count();
        let rank_bits: Vec<bool> = (0..rank_len)
            .map(|_| !xorshift(&mut state).is_multiple_of(3))
            .collect();
        let rank_buffer = BitBuffer::from(rank_bits.clone());
        let rank_indices: Vec<usize> = rank_bits
            .iter()
            .enumerate()
            .filter_map(|(idx, &value)| value.then_some(idx))
            .collect();

        let expected = expected_intersect_by_rank(&base_bits, &rank_bits);

        assert_eq!(
            intersect_bit_buffers::<D>(&base_buffer, &rank_buffer, rank_indices.len()),
            expected
        );
        assert_eq!(
            intersect_bit_buffer_by_rank_indices::<D>(&base_buffer, &rank_indices),
            expected
        );
    }

    fn expected_intersect_by_rank(base_bits: &[bool], rank_bits: &[bool]) -> Mask {
        let mut rank = 0usize;
        Mask::from_iter(base_bits.iter().map(|&is_set| {
            if is_set {
                let keep = rank_bits[rank];
                rank += 1;
                keep
            } else {
                false
            }
        }))
    }
}
