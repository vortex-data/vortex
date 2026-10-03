// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

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
use crate::coder::train_bins;
use crate::decode::ChunkDecoder;
use crate::decode::IDS_SCRATCH;
use crate::decode::OutInt;
use crate::decode::decode_block;
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
    ptype: PType,
    unsliced_n_rows: usize,
    slice_start: usize,
    slice_stop: usize,
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
        2
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        if idx == 0 {
            BufferHandle::new_host(array.block_starts.clone())
        } else {
            BufferHandle::new_host(array.data.clone())
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        Some(if idx == 0 { "block_starts" } else { "data" }.to_string())
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(
            buffers.len() == 2,
            "expected 2 buffers, got {}",
            buffers.len()
        );
        let mut data = array.data().clone();
        data.block_starts = buffers[0].clone().try_to_host_sync()?;
        data.data = buffers[1].clone().try_to_host_sync()?;
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
            buffers.len() == 2,
            "expected 2 buffers, got {}",
            buffers.len()
        );
        let data = EntropyBinsData {
            metadata,
            block_starts: buffers[0].clone().try_to_host_sync()?,
            data: buffers[1].clone().try_to_host_sync()?,
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
    ) -> VortexResult<EntropyBinsArray> {
        let dtype = parray.dtype().clone();
        let validity = parray.validity()?;
        let data = EntropyBinsData::encode(parray, level)?;
        Self::try_new(dtype, data, validity)
    }
}

/// Integers mapped to order-preserving `u64` latents.
trait Latent: NativePType + OutInt + Into<PValue> {
    fn latent(self) -> u64;
}

macro_rules! latent_impl {
    ($signed:literal: $($t:ty),*) => {$(
        impl Latent for $t {
            // Sign-extend to 64 bits and flip the sign bit, so order is preserved.
            #[allow(clippy::cast_sign_loss, clippy::cast_lossless)]
            fn latent(self) -> u64 {
                if $signed { (self as i64 as u64) ^ SIGN } else { self as u64 }
            }
        }
    )*};
}
latent_impl!(false: u8, u16, u32, u64);
latent_impl!(true: i8, i16, i32, i64);

impl EntropyBinsData {
    fn encode(parray: ArrayView<'_, Primitive>, level: usize) -> VortexResult<Self> {
        let ptype = parray.ptype();
        vortex_ensure!(ptype.is_int(), "entropy bins encode integers, got {ptype}");
        let latents: Vec<u64> = match_each_integer_ptype!(ptype, |T| {
            parray.as_slice::<T>().iter().map(|&v| v.latent()).collect()
        });
        let n = latents.len();
        let mut metadata = EntropyBinsMetadata::default();
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
            metadata,
            block_starts: ByteBuffer::from(block_starts),
            data: ByteBuffer::from(data),
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
        if self.ptype.is_signed_int() { SIGN } else { 0 }
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
        let base = self.base();
        let blocks_per_chunk = CHUNK_VALUES / BLOCK_VALUES;
        let mut ids = vec![0u8; 4 * IDS_SCRATCH];
        let data = self.data.as_slice();
        let mut b = first;
        while b <= last {
            let ci = b / blocks_per_chunk;
            let chunk = self
                .metadata
                .chunks
                .get(ci)
                .ok_or_else(|| vortex_err!("missing chunk {ci}"))?;
            let decoder = ChunkDecoder::new(chunk, base)?;
            let chunk_last = ((ci + 1) * blocks_per_chunk - 1).min(last);
            while b <= chunk_last {
                let row0 = b * BLOCK_VALUES;
                let n = BLOCK_VALUES.min(self.unsliced_n_rows - row0);
                let view = parse_block(data, self.block_start(b), n, decoder.table.as_ref())?;
                let dst = &mut out[row0 - first * BLOCK_VALUES..row0 - first * BLOCK_VALUES + n];
                decode_block(&decoder, &view, &mut ids[..IDS_SCRATCH], dst);
                b += 1;
            }
        }
        let offset = start - first * BLOCK_VALUES;
        Ok(out.freeze().slice(offset..offset + (stop - start)))
    }

    /// Decode one row by decoding its block's ids up to that row and reading one offset.
    fn scalar_at<T: NativePType + OutInt + Into<PValue>>(&self, row: usize) -> VortexResult<T> {
        let block = row / BLOCK_VALUES;
        let pos = row % BLOCK_VALUES;
        let ci = row / CHUNK_VALUES;
        let chunk = self
            .metadata
            .chunks
            .get(ci)
            .ok_or_else(|| vortex_err!("missing chunk {ci}"))?;
        let decoder = ChunkDecoder::new(chunk, self.base())?;
        let len = BLOCK_VALUES.min(self.unsliced_n_rows - block * BLOCK_VALUES);
        let view = parse_block(
            self.data.as_slice(),
            self.block_start(block),
            len,
            decoder.table.as_ref(),
        )?;
        let mut ids = [0u8; IDS_SCRATCH];
        decoder.ids(&view, &mut ids, pos + 1);
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
