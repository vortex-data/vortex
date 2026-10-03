// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::OnceLock;

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::TypedArrayRef;
use vortex_array::array_slots;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::EntropyBinsMetadata;
use crate::coder::BLOCK_VALUES;
use crate::coder::CHUNK_VALUES;
use crate::coder::IdTable;
use crate::coder::TAIL_PADDING;
use crate::coder::encode_block;
use crate::coder::held_out_bits;
use crate::coder::train_bins;
use crate::decode::BlockView;
use crate::decode::ChunkDecoder;
use crate::decode::IDS_SCRATCH;
use crate::decode::OutInt;
use crate::decode::decode_ids;
use crate::decode::merge_block;
use crate::decode::parse_block;

const SIGN: u64 = 1 << 63;

/// An [`EntropyBins`]-encoded Vortex array.
pub type EntropyBinsArray = Array<EntropyBins>;

/// Entropy-coded bins array encoding marker.
#[derive(Clone, Debug)]
pub struct EntropyBins;

#[array_slots(EntropyBins)]
pub struct EntropyBinsSlots {
    /// The validity bitmap indicating which elements are non-null.
    #[slot(0)]
    pub validity: Option<ArrayRef>,
}

/// Additional typed accessors for entropy-bins arrays.
pub trait EntropyBinsArrayExt: EntropyBinsArraySlotsExt {
    /// Reconstruct the unsliced [`Validity`] from the validity slot.
    fn unsliced_validity(&self) -> Validity {
        child_to_validity(
            self.as_ref().slots()[EntropyBinsSlots::VALIDITY].as_ref(),
            self.as_ref().dtype().nullability(),
        )
    }
}
impl<T: TypedArrayRef<EntropyBins>> EntropyBinsArrayExt for T {}

/// Encoding-specific data for an [`EntropyBinsArray`].
#[derive(Clone, Debug)]
pub struct EntropyBinsData {
    pub(crate) metadata: EntropyBinsMetadata,
    /// Byte offset of every block in `data`, plus the end of the last block (u32 little-endian).
    pub(crate) block_starts: ByteBuffer,
    /// Block segments followed by [`TAIL_PADDING`] zero bytes.
    pub(crate) data: ByteBuffer,
    /// With delta coding, each block's first value as `ptype` bytes (little-endian); else empty.
    pub(crate) seeds: ByteBuffer,
    ptype: PType,
    unsliced_n_rows: usize,
    slice_start: usize,
    slice_stop: usize,
    /// Decode tables per chunk, built on first use and shared by slices and clones.
    decoders: Arc<[OnceLock<ChunkDecoder>]>,
}

fn empty_decoders(n_chunks: usize) -> Arc<[OnceLock<ChunkDecoder>]> {
    (0..n_chunks).map(|_| OnceLock::new()).collect()
}

impl Display for EntropyBinsData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ptype: {}, nrows: {}, slice: {}..{}",
            self.ptype, self.unsliced_n_rows, self.slice_start, self.slice_stop
        )
    }
}

impl ArrayHash for EntropyBinsData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.unsliced_n_rows.hash(state);
        self.slice_start.hash(state);
        self.slice_stop.hash(state);
        self.metadata.encode_to_vec().hash(state);
        self.block_starts.array_hash(state, accuracy);
        self.data.array_hash(state, accuracy);
        self.seeds.array_hash(state, accuracy);
    }
}

impl ArrayEq for EntropyBinsData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.unsliced_n_rows == other.unsliced_n_rows
            && self.slice_start == other.slice_start
            && self.slice_stop == other.slice_stop
            && self.metadata == other.metadata
            && self.block_starts.array_eq(&other.block_starts, accuracy)
            && self.data.array_eq(&other.data, accuracy)
            && self.seeds.array_eq(&other.seeds, accuracy)
    }
}

impl VTable for EntropyBins {
    type TypedArrayData = EntropyBinsData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.entropy_bins");
        *ID
    }

    fn validate(
        &self,
        data: &EntropyBinsData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let validity = child_to_validity(
            EntropyBinsSlotsView::from_slots(slots).validity,
            dtype.nullability(),
        );
        data.validate(dtype, len, &validity)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        3
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        BufferHandle::new_host(match idx {
            0 => array.block_starts.clone(),
            1 => array.data.clone(),
            _ => array.seeds.clone(),
        })
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        Some(
            match idx {
                0 => "block_starts",
                1 => "data",
                _ => "seeds",
            }
            .to_string(),
        )
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(
            buffers.len() == 3,
            "expected 3 buffers, got {}",
            buffers.len()
        );
        let mut data = array.data().clone();
        data.block_starts = buffers[0].clone().try_to_host_sync()?;
        data.data = buffers[1].clone().try_to_host_sync()?;
        data.seeds = buffers[2].clone().try_to_host_sync()?;
        Ok(
            ArrayParts::new(self.clone(), array.dtype().clone(), array.len(), data)
                .with_slots(array.slots().iter().cloned().collect()),
        )
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(array.metadata.encode_to_vec()))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = EntropyBinsMetadata::decode(metadata)?;
        let validity = if children.is_empty() {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            Validity::Array(children.get(0, &Validity::DTYPE, len)?)
        } else {
            vortex_bail!(
                "EntropyBinsArray expected 0 or 1 child, got {}",
                children.len()
            );
        };
        vortex_ensure!(
            buffers.len() == 3,
            "expected 3 buffers, got {}",
            buffers.len()
        );
        let data = EntropyBinsData {
            decoders: empty_decoders(metadata.chunks.len()),
            metadata,
            block_starts: buffers[0].clone().try_to_host_sync()?,
            data: buffers[1].clone().try_to_host_sync()?,
            seeds: buffers[2].clone().try_to_host_sync()?,
            ptype: dtype.as_ptype(),
            unsliced_n_rows: len,
            slice_start: 0,
            slice_stop: len,
        };
        let slots = EntropyBinsSlots {
            validity: validity_to_child(&validity, len),
        }
        .into_slots();
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        EntropyBinsSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let validity = array
            .unsliced_validity()
            .slice(array.data().slice_start..array.data().slice_stop)?;
        Ok(ExecutionResult::done(
            array.data().decompress(validity)?.into_array(),
        ))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        crate::rules::RULES.evaluate(array, parent, child_idx)
    }
}

impl EntropyBins {
    /// Constructs an array after validating its data and validity invariants.
    pub fn try_new(
        dtype: DType,
        data: EntropyBinsData,
        validity: Validity,
    ) -> VortexResult<EntropyBinsArray> {
        let len = data.len();
        data.validate(&dtype, len, &validity)?;
        let slots = EntropyBinsSlots {
            validity: validity_to_child(&validity, data.unsliced_n_rows),
        }
        .into_slots();
        Array::try_from_parts(ArrayParts::new(EntropyBins, dtype, len, data).with_slots(slots))
    }

    /// Compress an integer primitive array. Null rows are encoded with whatever value the
    /// buffer holds; validity is kept as a child.
    pub fn from_primitive(
        parray: ArrayView<'_, Primitive>,
        level: usize,
        delta: bool,
    ) -> VortexResult<EntropyBinsArray> {
        let dtype = parray.dtype().clone();
        let validity = parray.validity()?;
        let data = EntropyBinsData::encode(parray, level, delta)?;
        Self::try_new(dtype, data, validity)
    }
}

impl EntropyBins {
    /// Estimate the encoded size in bytes without encoding: bins are trained on every other
    /// sampled block and scored on the blocks in between, plus the per-block and per-array
    /// overheads.
    pub fn estimate_nbytes(
        parray: ArrayView<'_, Primitive>,
        level: usize,
        delta: bool,
    ) -> VortexResult<usize> {
        let ptype = parray.ptype();
        vortex_ensure!(ptype.is_int(), "entropy bins encode integers, got {ptype}");
        let latents = latents_of(parray, delta);
        let n = latents.len();
        // Up to 32 evenly spaced units, alternating between training and scoring.
        let unit = if n >= 8 * BLOCK_VALUES {
            BLOCK_VALUES
        } else {
            (n / 8).max(1)
        };
        let n_units = (n / unit).clamp(1, 32);
        let (mut train, mut test) = (Vec::new(), Vec::new());
        for i in 0..n_units {
            let start = if n_units == 1 {
                0
            } else {
                i * (n - unit) / (n_units - 1)
            };
            let dst = if i % 2 == 0 { &mut train } else { &mut test };
            dst.extend_from_slice(&latents[start..(start + unit).min(n)]);
        }
        if test.is_empty() {
            test.clone_from(&train);
        }
        let bits = held_out_bits(&train, &test, level)?;
        // Per block: header, lane states, offset, part-filled lane words.
        let per_block = 1 + 2 + 14 + 4 + 16;
        let n_blocks = n.div_ceil(BLOCK_VALUES);
        let fixed = TAIL_PADDING + 64 * 14 * n.div_ceil(CHUNK_VALUES);
        let seeds = if delta {
            ptype.byte_width() * n_blocks
        } else {
            0
        };
        // A size estimate: rounding the fractional bytes down is fine.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let coded = (bits * n as f64 / 8.0) as usize;
        let estimate = coded + per_block * n_blocks + fixed + seeds;
        Ok(estimate)
    }
}

/// Integers mapped to order-preserving `u64` latents.
trait Latent: NativePType + OutInt + Into<PValue> {
    fn latent(self) -> u64;
    fn wide(self) -> u64;
}

macro_rules! latent_impl {
    ($signed:literal: $($t:ty),*) => {$(
        impl Latent for $t {
            // Sign-extend to 64 bits and flip the sign bit, so order is preserved.
            #[allow(clippy::cast_sign_loss, clippy::cast_lossless)]
            fn latent(self) -> u64 {
                if $signed { (self as i64 as u64) ^ SIGN } else { self as u64 }
            }
            #[allow(clippy::cast_sign_loss, clippy::cast_lossless)]
            fn wide(self) -> u64 {
                if $signed { self as i64 as u64 } else { self as u64 }
            }
        }
    )*};
}
latent_impl!(false: u8, u16, u32, u64);
latent_impl!(true: i8, i16, i32, i64);

/// The value of every row as a 64-bit integer (sign- or zero-extended).
fn wide_values(parray: ArrayView<'_, Primitive>) -> Vec<u64> {
    match_each_integer_ptype!(parray.ptype(), |T| {
        parray.as_slice::<T>().iter().map(|&v| v.wide()).collect()
    })
}

/// The latents to code: order-preserving values, or per-block deltas whose first slot holds
/// the block's next delta (the seed carries the first value, so that slot only needs to be
/// cheap to code).
fn latents_of(parray: ArrayView<'_, Primitive>, delta: bool) -> Vec<u64> {
    if !delta {
        return match_each_integer_ptype!(parray.ptype(), |T| {
            parray.as_slice::<T>().iter().map(|&v| v.latent()).collect()
        });
    }
    let wide = wide_values(parray);
    let mut out = Vec::with_capacity(wide.len());
    for block in wide.chunks(BLOCK_VALUES) {
        let deltas: Vec<u64> = block
            .windows(2)
            .map(|w| w[1].wrapping_sub(w[0]) ^ SIGN)
            .collect();
        out.push(deltas.first().copied().unwrap_or(SIGN));
        out.extend(deltas);
    }
    out
}

impl EntropyBinsData {
    fn encode(parray: ArrayView<'_, Primitive>, level: usize, delta: bool) -> VortexResult<Self> {
        let ptype = parray.ptype();
        vortex_ensure!(ptype.is_int(), "entropy bins encode integers, got {ptype}");
        let latents = latents_of(parray, delta);
        let seeds: Vec<u8> = if delta {
            let width = ptype.byte_width();
            wide_values(parray)
                .chunks(BLOCK_VALUES)
                .flat_map(|b| b[0].to_le_bytes()[..width].to_vec())
                .collect()
        } else {
            Vec::new()
        };
        let n = latents.len();
        let mut metadata = EntropyBinsMetadata {
            delta,
            ..Default::default()
        };
        let mut data = Vec::new();
        let mut starts: Vec<u32> = Vec::with_capacity(n.div_ceil(BLOCK_VALUES) + 1);
        for chunk_latents in latents.chunks(CHUNK_VALUES) {
            let chunk = train_bins(chunk_latents, level)?;
            let table = IdTable::new(&chunk)?;
            for block in chunk_latents.chunks(BLOCK_VALUES) {
                starts.push(u32::try_from(data.len())?);
                encode_block(&chunk, table.as_ref(), block, &mut data)?;
            }
            metadata.chunks.push(chunk);
        }
        starts.push(u32::try_from(data.len())?);
        data.resize(data.len() + TAIL_PADDING, 0);
        let block_starts: Vec<u8> = starts.iter().flat_map(|s| s.to_le_bytes()).collect();
        Ok(Self {
            decoders: empty_decoders(metadata.chunks.len()),
            metadata,
            block_starts: ByteBuffer::from(block_starts),
            data: ByteBuffer::from(data),
            seeds: ByteBuffer::from(seeds),
            ptype,
            unsliced_n_rows: n,
            slice_start: 0,
            slice_stop: n,
        })
    }

    /// Validate dtype, validity, slice and block invariants.
    pub fn validate(&self, dtype: &DType, len: usize, validity: &Validity) -> VortexResult<()> {
        vortex_ensure!(
            self.ptype.is_int(),
            "expected an integer ptype, got {}",
            self.ptype
        );
        vortex_ensure!(
            dtype.as_ptype() == self.ptype,
            "expected ptype {}, got {}",
            self.ptype,
            dtype.as_ptype()
        );
        vortex_ensure!(
            dtype.nullability() == validity.nullability(),
            "nullability mismatch"
        );
        vortex_ensure!(
            self.slice_start <= self.slice_stop
                && self.slice_stop <= self.unsliced_n_rows
                && self.slice_stop - self.slice_start == len,
            "invalid slice {}..{} of {} rows for len {len}",
            self.slice_start,
            self.slice_stop,
            self.unsliced_n_rows
        );
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == self.unsliced_n_rows,
                "validity length mismatch"
            );
        }
        let n_values: usize = self
            .metadata
            .chunks
            .iter()
            .map(|c| c.n_values as usize)
            .sum();
        vortex_ensure!(
            n_values == self.unsliced_n_rows,
            "chunk sizes do not cover the array"
        );
        for (i, c) in self.metadata.chunks.iter().enumerate() {
            vortex_ensure!(
                c.lowers.len() == c.widths.len() && c.widths.len() == c.weights.len(),
                "chunk {i} has inconsistent bins"
            );
            vortex_ensure!(
                c.widths.iter().all(|&w| w <= 64),
                "chunk {i} has a bin wider than 64 bits"
            );
            if i + 1 < self.metadata.chunks.len() {
                vortex_ensure!(c.n_values as usize == CHUNK_VALUES, "chunk {i} is short");
            }
        }
        let n_blocks = self.unsliced_n_rows.div_ceil(BLOCK_VALUES);
        vortex_ensure!(
            self.block_starts.len() == 4 * (n_blocks + 1),
            "expected {} block offsets",
            n_blocks + 1
        );
        let expected_seeds = if self.metadata.delta {
            n_blocks * self.ptype.byte_width()
        } else {
            0
        };
        vortex_ensure!(
            self.seeds.len() == expected_seeds,
            "expected {expected_seeds} seed bytes, got {}",
            self.seeds.len()
        );
        let end = self.block_start(n_blocks);
        vortex_ensure!(
            end + TAIL_PADDING <= self.data.len(),
            "data buffer lacks its tail padding"
        );
        Ok(())
    }

    fn block_start(&self, b: usize) -> usize {
        let s = &self.block_starts[4 * b..4 * b + 4];
        u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize
    }

    fn base(&self) -> u64 {
        if self.metadata.delta || self.ptype.is_signed_int() {
            SIGN
        } else {
            0
        }
    }

    /// The seed of block `b` with delta coding.
    fn seed(&self, b: usize) -> Option<u64> {
        if !self.metadata.delta {
            return None;
        }
        let width = self.ptype.byte_width();
        let mut bytes = [0u8; 8];
        bytes[..width].copy_from_slice(&self.seeds[b * width..(b + 1) * width]);
        Some(u64::from_le_bytes(bytes))
    }

    /// The decode tables of chunk `ci`, built once.
    fn decoder(&self, ci: usize) -> VortexResult<&ChunkDecoder> {
        let slot = self
            .decoders
            .get(ci)
            .ok_or_else(|| vortex_err!("missing chunk {ci}"))?;
        if let Some(d) = slot.get() {
            return Ok(d);
        }
        let chunk = self
            .metadata
            .chunks
            .get(ci)
            .ok_or_else(|| vortex_err!("missing chunk {ci}"))?;
        // A concurrent initialization may win; both build the same tables.
        drop(slot.set(ChunkDecoder::new(chunk, self.base())?));
        slot.get()
            .ok_or_else(|| vortex_err!("chunk {ci} decoder was not initialized"))
    }

    /// Decode the rows `slice_start..slice_stop`.
    pub(crate) fn decompress(&self, validity: Validity) -> VortexResult<PrimitiveArray> {
        match_each_integer_ptype!(self.ptype, |T| {
            let buffer = self.decode_range::<T>(self.slice_start, self.slice_stop)?;
            Ok(PrimitiveArray::new(buffer, validity))
        })
    }

    fn decode_range<T: NativePType + OutInt>(
        &self,
        start: usize,
        stop: usize,
    ) -> VortexResult<vortex_buffer::Buffer<T>> {
        if start == stop {
            return Ok(vortex_buffer::Buffer::empty());
        }
        let first = start / BLOCK_VALUES;
        let last = (stop - 1) / BLOCK_VALUES;
        let covered = (last + 1) * BLOCK_VALUES - first * BLOCK_VALUES;
        let mut out = BufferMut::<T>::with_capacity(covered);
        // SAFETY: every position of the covered blocks is written below before it is read.
        unsafe { out.set_len(covered.min(self.unsliced_n_rows - first * BLOCK_VALUES)) };
        let blocks_per_chunk = CHUNK_VALUES / BLOCK_VALUES;
        let mut ids = vec![0u8; 4 * IDS_SCRATCH];
        let data = self.data.as_slice();
        let mut b = first;
        while b <= last {
            let ci = b / blocks_per_chunk;
            let decoder = self.decoder(ci)?;
            let chunk_last = ((ci + 1) * blocks_per_chunk - 1).min(last);
            while b <= chunk_last {
                let group = if b + 3 <= chunk_last && (b + 4) * BLOCK_VALUES <= self.unsliced_n_rows
                {
                    4
                } else {
                    1
                };
                let views = (0..group)
                    .map(|k| {
                        let n = BLOCK_VALUES.min(self.unsliced_n_rows - (b + k) * BLOCK_VALUES);
                        parse_block(data, self.block_start(b + k), n, decoder.table.as_ref())
                    })
                    .collect::<VortexResult<Vec<_>>>()?;
                decode_ids(decoder, &views, &mut ids);
                for (k, view) in views.iter().enumerate() {
                    let row0 = (b + k - first) * BLOCK_VALUES;
                    merge_block(
                        decoder,
                        view,
                        &ids[k * IDS_SCRATCH..(k + 1) * IDS_SCRATCH],
                        &mut out[row0..row0 + view.n],
                        self.seed(b + k),
                    );
                }
                b += group;
            }
        }
        let offset = start - first * BLOCK_VALUES;
        Ok(out.freeze().slice(offset..offset + (stop - start)))
    }

    /// Decode one row by decoding its block's ids up to that row and reading one offset.
    fn scalar_at<T: NativePType + OutInt + Into<PValue>>(&self, row: usize) -> VortexResult<T> {
        let block = row / BLOCK_VALUES;
        let pos = row % BLOCK_VALUES;
        let decoder = self.decoder(row / CHUNK_VALUES)?;
        let len = BLOCK_VALUES.min(self.unsliced_n_rows - block * BLOCK_VALUES);
        let view = parse_block(
            self.data.as_slice(),
            self.block_start(block),
            len,
            decoder.table.as_ref(),
        )?;
        let mut ids = [0u8; IDS_SCRATCH];
        decoder.ids(&view, &mut ids, pos + 1);
        if self.metadata.delta {
            // Deltas need the running sum up to the row: merge the block's prefix.
            let mut tmp = [T::default(); BLOCK_VALUES];
            let prefix = BlockView { n: pos + 1, ..view };
            merge_block(decoder, &prefix, &ids, &mut tmp, self.seed(block));
            return Ok(tmp[pos]);
        }
        Ok(T::truncate_from(decoder.value_at(&view, &ids, pos)))
    }

    pub(crate) fn sliced(&self, start: usize, stop: usize) -> Self {
        Self {
            slice_start: self.slice_start + start,
            slice_stop: self.slice_start + stop,
            ..self.clone()
        }
    }

    /// Returns the number of elements in the array.
    pub fn len(&self) -> usize {
        self.slice_stop - self.slice_start
    }

    /// Returns `true` if the array contains no elements.
    pub fn is_empty(&self) -> bool {
        self.slice_stop == self.slice_start
    }
}

impl ValidityVTable<EntropyBins> for EntropyBins {
    fn validity(array: ArrayView<'_, EntropyBins>) -> VortexResult<Validity> {
        array
            .unsliced_validity()
            .slice(array.data().slice_start..array.data().slice_stop)
    }
}

impl OperationsVTable<EntropyBins> for EntropyBins {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, EntropyBins>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let data = array.data();
        let row = data.slice_start + index;
        let nullability = array.dtype().nullability();
        match_each_integer_ptype!(data.ptype, |T| {
            Ok(Scalar::primitive(data.scalar_at::<T>(row)?, nullability))
        })
    }
}
