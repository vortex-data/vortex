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
use vortex_array::ProbeState;
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
use crate::coder::MAX_BLOCK_VALUES;
use crate::coder::MAX_LAG;
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
    /// Byte length of every block in `data` (u16 little-endian).
    pub(crate) block_lengths: ByteBuffer,
    /// Block segments followed by [`TAIL_PADDING`] zero bytes.
    pub(crate) data: ByteBuffer,
    /// With `lag > 0`, each block's first `lag` values as `ptype` bytes (little-endian, zero for
    /// rows past the end); else empty.
    pub(crate) seeds: ByteBuffer,
    ptype: PType,
    unsliced_n_rows: usize,
    slice_start: usize,
    slice_stop: usize,
    /// Decode tables per chunk, built on first use and shared by slices and clones.
    decoders: Arc<[OnceLock<ChunkDecoder>]>,
    /// Byte offset of every block in `data` plus the end of the last, from `block_lengths`;
    /// built on first use and shared by slices and clones.
    starts: Arc<OnceLock<Vec<u32>>>,
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
        self.block_lengths.array_hash(state, accuracy);
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
            && self.block_lengths.array_eq(&other.block_lengths, accuracy)
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
            0 => array.block_lengths.clone(),
            1 => array.data.clone(),
            _ => array.seeds.clone(),
        })
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        Some(
            match idx {
                0 => "block_lengths",
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
        data.block_lengths = buffers[0].clone().try_to_host_sync()?;
        data.starts = Arc::default();
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
            block_lengths: buffers[0].clone().try_to_host_sync()?,
            starts: Arc::default(),
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

    /// Compress an integer primitive array with the given layout options. Null rows are encoded
    /// with whatever value the buffer holds; validity is kept as a child.
    pub fn from_primitive(
        parray: ArrayView<'_, Primitive>,
        level: usize,
        options: EntropyBinsOptions,
    ) -> VortexResult<EntropyBinsArray> {
        let dtype = parray.dtype().clone();
        let validity = parray.validity()?;
        let data = EntropyBinsData::encode(parray, level, options)?;
        Self::try_new(dtype, data, validity)
    }

    /// Choose the layout for `parray` under `config` and estimate the encoded size, without
    /// encoding. Bins are trained on every other sampled block and scored on the blocks in
    /// between; per-block and per-array overheads are added on top.
    pub fn plan(
        parray: ArrayView<'_, Primitive>,
        config: &EntropyBinsConfig,
    ) -> VortexResult<EntropyBinsPlan> {
        let ptype = parray.ptype();
        vortex_ensure!(ptype.is_int(), "entropy bins encode integers, got {ptype}");
        let n = parray.len();
        let units = sample_units(parray);
        let mut best: Option<(usize, usize)> = None;
        for &lag in config.lags {
            let coded = estimate_coded(&units, n, ptype, config.level, lag)?;
            if best.is_none_or(|(_, b)| coded < b) {
                best = Some((lag, coded));
            }
        }
        let (lag, coded) = best.ok_or_else(|| vortex_err!("no lags to plan"))?;
        let fixed = TAIL_PADDING + 64 * 14 * n.div_ceil(CHUNK_VALUES);
        let total = |block_values: usize| {
            let n_blocks = n.div_ceil(block_values);
            coded + fixed + n_blocks * (PER_BLOCK_BYTES + lag * ptype.byte_width())
        };
        // Larger blocks only when they save a noticeable share: they slow random access.
        let max_block = config
            .max_block_values
            .clamp(BLOCK_VALUES, MAX_BLOCK_VALUES)
            .next_power_of_two()
            .min(MAX_BLOCK_VALUES);
        let floor = total(max_block);
        let block_values = [BLOCK_VALUES, 2 * BLOCK_VALUES, MAX_BLOCK_VALUES]
            .into_iter()
            .filter(|&b| b <= max_block)
            .find(|&b| total(b) * 100 <= floor * (100 + config.larger_block_gain_percent))
            .unwrap_or(max_block);
        let nbytes = total(block_values);
        // 8-bit refill words halve the bits lanes leave unused at the end of a block, at a few
        // percent of decode speed: worth it where that is a noticeable share of the bytes.
        let saved = n.div_ceil(block_values) * NARROW_WORD_SAVING_BYTES;
        let narrow_words = config
            .narrow_word_gain_percent
            .is_some_and(|percent| saved * 100 >= nbytes * percent);
        let (word_bits, nbytes) = if narrow_words {
            (8, nbytes - saved)
        } else {
            (16, nbytes)
        };
        Ok(EntropyBinsPlan {
            options: EntropyBinsOptions::new(lag, block_values).with_word_bits(word_bits),
            nbytes,
        })
    }
}

/// Dials for [`EntropyBins::plan`], trading size against decode speed, random access and
/// compression time. The presets are measured starting points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntropyBinsConfig {
    /// pco compression level used to train the bins (0 to 12). Lower levels compress faster and
    /// find fewer, coarser bins: level 4 compresses ~25% faster for ~7% more bytes.
    pub level: usize,
    /// Row distances tried for difference coding (zero codes the values). Each costs one
    /// estimate at compression time.
    pub lags: &'static [usize],
    /// Largest block size the planner may choose, from [`BLOCK_VALUES`] to [`MAX_BLOCK_VALUES`].
    /// Random access decodes up to a block, so this bounds its cost.
    pub max_block_values: usize,
    /// How much smaller (in percent) the largest allowed blocks must make the array before a
    /// larger block size is chosen.
    pub larger_block_gain_percent: usize,
    /// How much smaller (in percent) 8-bit refill words must make the array to be chosen, or
    /// `None` to always use 16-bit words (the faster decode).
    pub narrow_word_gain_percent: Option<usize>,
}

impl EntropyBinsConfig {
    /// The default: larger blocks and 8-bit words only where they clearly pay off.
    pub const BALANCED: Self = Self {
        level: 8,
        lags: &[0, 1, 2, 3, 4, 8],
        max_block_values: MAX_BLOCK_VALUES,
        larger_block_gain_percent: 15,
        narrow_word_gain_percent: Some(1),
    };

    /// Fastest decode and random access: 1024-value blocks, 16-bit words, lag 0 or 1 only.
    pub const FAST: Self = Self {
        level: 8,
        lags: &[0, 1],
        max_block_values: BLOCK_VALUES,
        larger_block_gain_percent: 0,
        narrow_word_gain_percent: None,
    };

    /// Smallest output: larger blocks and 8-bit words wherever they save anything.
    pub const SMALLEST: Self = Self {
        level: 8,
        lags: &[0, 1, 2, 3, 4, 8],
        max_block_values: MAX_BLOCK_VALUES,
        larger_block_gain_percent: 1,
        narrow_word_gain_percent: Some(0),
    };
}

impl Default for EntropyBinsConfig {
    fn default() -> Self {
        Self::BALANCED
    }
}

/// How an array is laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntropyBinsOptions {
    /// Row distance of the coded differences, at most [`MAX_LAG`]; zero codes the values.
    pub lag: usize,
    /// Values per block: a power of two from [`BLOCK_VALUES`] to [`MAX_BLOCK_VALUES`].
    pub block_values: usize,
    /// Bits per tANS refill word: 8 or 16.
    pub word_bits: u32,
}

impl EntropyBinsOptions {
    /// Options with 16-bit refill words.
    pub const fn new(lag: usize, block_values: usize) -> Self {
        Self {
            lag,
            block_values,
            word_bits: 16,
        }
    }

    /// Use `word_bits`-bit refill words (8 or 16).
    #[must_use]
    pub const fn with_word_bits(mut self, word_bits: u32) -> Self {
        self.word_bits = word_bits;
        self
    }
}

/// An encoding choice and its estimated size, from [`EntropyBins::plan`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntropyBinsPlan {
    /// The chosen layout.
    pub options: EntropyBinsOptions,
    /// Estimated encoded size in bytes.
    pub nbytes: usize,
}

/// Measured fixed cost of a coded block with 16-bit refill words: header, lane states, the stop
/// field, part-filled lane words and the block's offset.
const PER_BLOCK_BYTES: usize = 37;

/// Measured bytes per block that 8-bit refill words save over 16-bit ones.
const NARROW_WORD_SAVING_BYTES: usize = 8;

/// The estimated bytes of the coded ids and offsets alone.
/// Up to 32 evenly spaced runs of rows (sign- or zero-extended), alternately for training and
/// scoring the estimate's bins. Only these rows are read.
fn sample_units(parray: ArrayView<'_, Primitive>) -> Vec<Vec<u64>> {
    let n = parray.len();
    let unit = if n >= 8 * BLOCK_VALUES {
        BLOCK_VALUES
    } else {
        (n / 8).max(1)
    };
    let n_units = (n / unit).clamp(1, 32);
    match_each_integer_ptype!(parray.ptype(), |T| {
        let values = parray.as_slice::<T>();
        (0..n_units)
            .map(|i| {
                let start = if n_units == 1 {
                    0
                } else {
                    i * (n - unit) / (n_units - 1)
                };
                values[start..(start + unit).min(n)]
                    .iter()
                    .map(|&v| v.wide())
                    .collect()
            })
            .collect()
    })
}

fn estimate_coded(
    units: &[Vec<u64>],
    n: usize,
    ptype: PType,
    level: usize,
    lag: usize,
) -> VortexResult<usize> {
    let transform = Transform::new(ptype.is_signed_int(), lag)?;
    let (mut train, mut test) = (Vec::new(), Vec::new());
    for (i, unit) in units.iter().enumerate() {
        let dst = if i % 2 == 0 { &mut train } else { &mut test };
        transform.extend_latents(unit, dst);
    }
    if test.is_empty() {
        test.clone_from(&train);
    }
    let bits = held_out_bits(&train, &test, level)?;
    // A size estimate: rounding the fractional bytes down is fine.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Ok((bits * n as f64 / 8.0) as usize)
}

/// The `block_log` of a supported block size.
fn block_log_of(block_values: usize) -> VortexResult<u32> {
    vortex_ensure!(
        block_values.is_power_of_two() && (BLOCK_VALUES..=MAX_BLOCK_VALUES).contains(&block_values),
        "block size {block_values} is not a power of two from {BLOCK_VALUES} to {MAX_BLOCK_VALUES}"
    );
    Ok(block_values.trailing_zeros())
}

/// How rows map to latents: order-preserving values with `lag == 0` (signed values with their
/// sign bit flipped), else the wrapping difference from the row `lag` back with its sign bit
/// flipped, so latents order like the signed differences.
#[derive(Clone, Copy, Debug)]
struct Transform {
    lag: usize,
    signed: bool,
}

impl Transform {
    fn new(signed: bool, lag: usize) -> VortexResult<Self> {
        vortex_ensure!(lag <= MAX_LAG, "lag {lag} exceeds {MAX_LAG}");
        Ok(Self { lag, signed })
    }

    /// Append the latents of one block. With a lag, the block's first `lag` slots repeat the
    /// latents `lag` rows later (the seeds carry those rows, so the slots only need to be cheap).
    fn extend_latents(&self, block: &[u64], out: &mut Vec<u64>) {
        if self.lag == 0 {
            let flip = if self.signed { SIGN } else { 0 };
            out.extend(block.iter().map(|&w| w ^ flip));
            return;
        }
        let lag = self.lag;
        let diff = |i: usize| block[i].wrapping_sub(block[i - lag]) ^ SIGN;
        for i in 0..lag.min(block.len()) {
            out.push(if i + lag < block.len() {
                diff(i + lag)
            } else {
                SIGN
            });
        }
        out.extend((lag..block.len()).map(diff));
    }

    /// What the decoder adds to every latent (wrapping) to undo the flipped sign bit.
    fn base(&self) -> u64 {
        if self.lag > 0 || self.signed { SIGN } else { 0 }
    }
}

/// Integers sign- or zero-extended to 64 bits.
pub(crate) trait Wide: NativePType {
    fn wide(self) -> u64;
}

macro_rules! wide_impl {
    ($signed:literal: $($t:ty),*) => {$(
        impl Wide for $t {
            #[allow(clippy::cast_sign_loss, clippy::cast_lossless)]
            fn wide(self) -> u64 {
                if $signed { self as i64 as u64 } else { self as u64 }
            }
        }
    )*};
}
wide_impl!(false: u8, u16, u32, u64);
wide_impl!(true: i8, i16, i32, i64);

impl EntropyBinsData {
    fn encode(
        parray: ArrayView<'_, Primitive>,
        level: usize,
        options: EntropyBinsOptions,
    ) -> VortexResult<Self> {
        let EntropyBinsOptions {
            lag,
            block_values,
            word_bits,
        } = options;
        let ptype = parray.ptype();
        vortex_ensure!(ptype.is_int(), "entropy bins encode integers, got {ptype}");
        let block_log = block_log_of(block_values)?;
        let transform = Transform::new(ptype.is_signed_int(), lag)?;
        let mut latents = Vec::with_capacity(parray.len());
        let mut seeds = Vec::new();
        let width = ptype.byte_width();
        // Widen one block at a time into a reused buffer rather than the whole array.
        let mut block = Vec::with_capacity(block_values);
        match_each_integer_ptype!(ptype, |T| {
            for values in parray.as_slice::<T>().chunks(block_values) {
                block.clear();
                block.extend(values.iter().map(|&v| v.wide()));
                transform.extend_latents(&block, &mut latents);
                for i in 0..lag {
                    let v = block.get(i).copied().unwrap_or(0);
                    seeds.extend_from_slice(&v.to_le_bytes()[..width]);
                }
            }
        });
        let n = latents.len();
        let mut metadata = EntropyBinsMetadata {
            chunks: Vec::new(),
            lag: u32::try_from(lag)?,
            block_log,
            word_bits,
        };
        let mut data = Vec::new();
        let mut lengths: Vec<u8> = Vec::with_capacity(2 * n.div_ceil(block_values));
        for chunk_latents in latents.chunks(CHUNK_VALUES) {
            let chunk = train_bins(chunk_latents, level)?;
            let table = IdTable::new(&chunk, word_bits)?;
            for block in chunk_latents.chunks(block_values) {
                let start = data.len();
                encode_block(&chunk, table.as_ref(), block, &mut data)?;
                lengths.extend_from_slice(&u16::try_from(data.len() - start)?.to_le_bytes());
            }
            metadata.chunks.push(chunk);
        }
        u32::try_from(data.len())?;
        data.resize(data.len() + TAIL_PADDING, 0);
        data.shrink_to_fit();
        Ok(Self {
            decoders: empty_decoders(metadata.chunks.len()),
            metadata,
            block_lengths: ByteBuffer::from(lengths),
            starts: Arc::default(),
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
        block_log_of(1 << self.metadata.block_log.min(31))?;
        vortex_ensure!(
            matches!(self.metadata.word_bits, 8 | 16),
            "refill words must be 8 or 16 bits, got {}",
            self.metadata.word_bits
        );
        let n_blocks = self.unsliced_n_rows.div_ceil(self.block_values());
        vortex_ensure!(
            self.block_lengths.len() == 2 * n_blocks,
            "expected {n_blocks} block lengths"
        );
        vortex_ensure!(
            self.lag() <= MAX_LAG,
            "lag {} exceeds {MAX_LAG}",
            self.metadata.lag
        );
        let expected_seeds = n_blocks * self.lag() * self.ptype.byte_width();
        vortex_ensure!(
            self.seeds.len() == expected_seeds,
            "expected {expected_seeds} seed bytes, got {}",
            self.seeds.len()
        );
        let end: u64 = self
            .block_lengths
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&l| u64::from(u16::from_le_bytes(l)))
            .sum();
        vortex_ensure!(
            end + TAIL_PADDING as u64 <= self.data.len() as u64,
            "data buffer lacks its tail padding"
        );
        vortex_ensure!(u32::try_from(end).is_ok(), "blocks span more than 4 GiB");
        Ok(())
    }

    pub(crate) fn block_start(&self, b: usize) -> usize {
        let starts = self.starts.get_or_init(|| {
            let mut at = 0u32;
            let mut starts = Vec::with_capacity(self.block_lengths.len() / 2 + 1);
            starts.push(0);
            for &l in self.block_lengths.as_chunks::<2>().0 {
                // `validate` checked that the lengths sum to at most `u32::MAX`.
                at += u32::from(u16::from_le_bytes(l));
                starts.push(at);
            }
            starts
        });
        starts[b] as usize
    }

    pub(crate) fn block_values(&self) -> usize {
        1 << self.metadata.block_log
    }

    pub(crate) fn ptype(&self) -> PType {
        self.ptype
    }

    /// The rows of the unsliced array this array covers.
    pub(crate) fn slice_range(&self) -> (usize, usize) {
        (self.slice_start, self.slice_stop)
    }

    pub(crate) fn unsliced_rows(&self) -> usize {
        self.unsliced_n_rows
    }

    fn lag(&self) -> usize {
        self.metadata.lag as usize
    }

    fn transform(&self) -> Transform {
        Transform {
            lag: self.lag(),
            signed: self.ptype.is_signed_int(),
        }
    }

    /// The seeds of block `b`: its first `lag` values, zero-extended (only their low `ptype`
    /// bits matter).
    pub(crate) fn seeds_of(&self, b: usize) -> ([u64; MAX_LAG], usize) {
        let lag = self.lag();
        let width = self.ptype.byte_width();
        let mut seeds = [0u64; MAX_LAG];
        let at = b * lag * width;
        for (i, seed) in seeds[..lag].iter_mut().enumerate() {
            let mut bytes = [0u8; 8];
            bytes[..width].copy_from_slice(&self.seeds[at + i * width..at + (i + 1) * width]);
            *seed = u64::from_le_bytes(bytes);
        }
        (seeds, lag)
    }

    /// The decode tables of chunk `ci`, built once.
    pub(crate) fn decoder(&self, ci: usize) -> VortexResult<&ChunkDecoder> {
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
        drop(slot.set(ChunkDecoder::new(
            chunk,
            self.transform().base(),
            self.metadata.word_bits,
        )?));
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

    pub(crate) fn decode_range<T: NativePType + OutInt>(
        &self,
        start: usize,
        stop: usize,
    ) -> VortexResult<vortex_buffer::Buffer<T>> {
        if start == stop {
            return Ok(vortex_buffer::Buffer::empty());
        }
        let bv = self.block_values();
        let first = start / bv;
        let last = (stop - 1) / bv;
        let covered = ((last + 1) * bv).min(self.unsliced_n_rows) - first * bv;
        let mut out = BufferMut::<T>::with_capacity(covered);
        // SAFETY: `decode_blocks` writes every position of the covered blocks.
        unsafe { out.set_len(covered) };
        self.decode_blocks(first, last + 1, &mut out, &mut [0u8; 4 * IDS_SCRATCH])?;
        let offset = start - first * bv;
        Ok(out.freeze().slice(offset..offset + (stop - start)))
    }

    /// Decode the blocks `first..stop` into `out`, which must hold exactly their rows. `ids` is
    /// scratch space for four blocks' ids.
    pub(crate) fn decode_blocks<T: OutInt>(
        &self,
        first: usize,
        stop: usize,
        out: &mut [T],
        ids: &mut [u8; 4 * IDS_SCRATCH],
    ) -> VortexResult<()> {
        let bv = self.block_values();
        let blocks_per_chunk = CHUNK_VALUES / bv;
        let data = self.data.as_slice();
        let mut b = first;
        while b < stop {
            let ci = b / blocks_per_chunk;
            let decoder = self.decoder(ci)?;
            let chunk_stop = ((ci + 1) * blocks_per_chunk).min(stop);
            while b < chunk_stop {
                let group = if b + 4 <= chunk_stop && (b + 4) * bv <= self.unsliced_n_rows {
                    4
                } else {
                    1
                };
                let views = (0..group)
                    .map(|k| {
                        let n = bv.min(self.unsliced_n_rows - (b + k) * bv);
                        parse_block(data, self.block_start(b + k), n, decoder.table.as_ref())
                    })
                    .collect::<VortexResult<Vec<_>>>()?;
                decode_ids(decoder, &views, ids);
                for (k, view) in views.iter().enumerate() {
                    let row0 = (b + k - first) * bv;
                    let (seeds, lag) = self.seeds_of(b + k);
                    merge_block(
                        decoder,
                        view,
                        &ids[k * IDS_SCRATCH..(k + 1) * IDS_SCRATCH],
                        &mut out[row0..row0 + view.n],
                        &seeds[..lag],
                    );
                }
                b += group;
            }
        }
        Ok(())
    }

    /// Decode one row by decoding its block's ids up to that row and reading one offset.
    fn scalar_at<T: NativePType + OutInt + Into<PValue>>(&self, row: usize) -> VortexResult<T> {
        let bv = self.block_values();
        let block = row / bv;
        let pos = row % bv;
        let decoder = self.decoder(row / CHUNK_VALUES)?;
        let len = bv.min(self.unsliced_n_rows - block * bv);
        let view = parse_block(
            self.data.as_slice(),
            self.block_start(block),
            len,
            decoder.table.as_ref(),
        )?;
        let mut ids = [0u8; IDS_SCRATCH];
        decoder.ids(&view, &mut ids, pos + 1);
        if self.lag() > 0 {
            // Differences need the running sums up to the row: merge the block's prefix.
            let mut tmp = [T::default(); MAX_BLOCK_VALUES];
            let prefix = BlockView { n: pos + 1, ..view };
            let (seeds, lag) = self.seeds_of(block);
            merge_block(decoder, &prefix, &ids, &mut tmp, &seeds[..lag]);
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

/// Retained across repeated lookups: the block touched last and, once it has been touched
/// twice, all of its values. A one-off lookup decodes only a block's prefix, while lookups that
/// stay within a block (binary searches, sorted takes) amortize one full block decode.
#[derive(Default)]
pub struct EntropyBinsProbeState {
    block: Option<usize>,
    /// The decoded block (sign- or zero-extended), empty until the second touch.
    values: Vec<u64>,
}

impl EntropyBinsData {
    fn probe_at<T: NativePType + OutInt + Wide + Into<PValue>>(
        &self,
        cache: &mut EntropyBinsProbeState,
        row: usize,
    ) -> VortexResult<T> {
        let bv = self.block_values();
        let block = row / bv;
        if cache.block != Some(block) {
            cache.block = Some(block);
            cache.values.clear();
            return self.scalar_at::<T>(row);
        }
        if cache.values.is_empty() {
            let start = block * bv;
            let stop = (start + bv).min(self.unsliced_n_rows);
            let decoded = self.decode_range::<T>(start, stop)?;
            cache.values.extend(decoded.iter().map(|&v| v.wide()));
        }
        Ok(T::truncate_from(cache.values[row % bv]))
    }
}

impl OperationsVTable<EntropyBins> for EntropyBins {
    type ProbeState = EntropyBinsProbeState;

    fn probe_scalar(
        state: &mut ProbeState<'_, EntropyBins>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        if !state.is_valid(index, ctx)? {
            return Ok(Scalar::null(state.array().dtype().clone()));
        }
        let array = state.array();
        let data = array.data();
        let row = data.slice_start + index;
        let nullability = array.dtype().nullability();
        match_each_integer_ptype!(data.ptype, |T| {
            let value = match state.retained() {
                Some(cache) => data.probe_at::<T>(cache, row)?,
                None => data.scalar_at::<T>(row)?,
            };
            Ok(Scalar::primitive(value, nullability))
        })
    }

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
