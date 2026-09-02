// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;

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
use vortex_array::arrays::slice::SliceReduce;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::dtype::half::f16;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::BlockResidualCodec;
use crate::BlockResidualParts;
use crate::codec::BlockResidualCodecEstimate;
use crate::codec::ResidualWord;
use crate::codec::packed_words_as_native;
use crate::codec::read_wide_bits;

const BLOCK_LEN: usize = 1024;
const METADATA_VERSION: u8 = 2;
const METADATA_LEN: usize = 41;

/// Ordered unsigned integers with one reference and packed residuals per block.
pub type BlockResidualArray = Array<BlockResidual>;

#[array_slots(BlockResidual)]
pub struct BlockResidualSlots {
    #[slot(0)]
    pub validity: Option<ArrayRef>,
}

#[derive(Clone, Debug)]
pub struct BlockResidualData {
    unsliced_len: usize,
    slice_start: usize,
    slice_stop: usize,
    payload: ByteBuffer,
    bases: Buffer<u64>,
    residual_widths: Buffer<u8>,
    high_widths: Buffer<u8>,
    residual_starts: Buffer<u32>,
    patch_starts: Buffer<u32>,
    high_starts: Buffer<u32>,
    residual_words: Buffer<u64>,
    patch_positions: Buffer<u16>,
    patch_highs: Buffer<u8>,
}

impl Display for BlockResidualData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "blocks: {}, slice: {}..{}",
            self.unsliced_len.div_ceil(BLOCK_LEN),
            self.slice_start,
            self.slice_stop
        )
    }
}

impl ArrayHash for BlockResidualData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.unsliced_len.hash(state);
        self.slice_start.hash(state);
        self.slice_stop.hash(state);
        self.payload.array_hash(state, accuracy);
    }
}

impl ArrayEq for BlockResidualData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.unsliced_len == other.unsliced_len
            && self.slice_start == other.slice_start
            && self.slice_stop == other.slice_stop
            && self.payload.array_eq(&other.payload, accuracy)
    }
}

#[derive(Clone, Debug)]
pub struct BlockResidual;

/// Exact encoded size and patch count without materialized payloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockResidualEstimate {
    nbytes: u64,
    patch_count: usize,
}

impl BlockResidualEstimate {
    pub(crate) fn try_new(
        encoded_nbytes: usize,
        validity_nbytes: u64,
        patch_count: usize,
    ) -> VortexResult<Self> {
        let nbytes = u64::try_from(encoded_nbytes)?
            .checked_add(validity_nbytes)
            .ok_or_else(|| vortex_error::vortex_err!("BlockResidual estimate size overflow"))?;
        Ok(Self {
            nbytes,
            patch_count,
        })
    }

    /// Return the estimated physical bytes.
    pub fn nbytes(self) -> u64 {
        self.nbytes
    }

    /// Return the estimated patch count.
    pub fn patch_count(self) -> usize {
        self.patch_count
    }
}

impl VTable for BlockResidual {
    type TypedArrayData = BlockResidualData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.block_residual");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let slots = BlockResidualSlotsView::from_slots(slots);
        let validity = child_to_validity(slots.validity, dtype.nullability());
        data.validate(dtype, len, slots, &validity)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        1
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => BufferHandle::new_host(array.payload.clone()),
            _ => vortex_panic!("BlockResidualArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some("payload".to_string()),
            _ => vortex_panic!("BlockResidualArray buffer_name {idx} out of bounds"),
        }
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(buffers.len() == 1, "BlockResidualArray expects one buffer");
        let mut data = array.data().clone();
        data.replace_payload(&buffers[0])?;
        Ok(
            ArrayParts::new(self.clone(), array.dtype().clone(), array.len(), data)
                .with_slots(array.slots().iter().cloned().collect()),
        )
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            BlockResidualMetadata::from_data(array.data())?.encode(),
        ))
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
        let metadata = BlockResidualMetadata::decode(metadata)?;
        let unsliced_len = usize::try_from(metadata.unsliced_len)?;
        let slice_start = usize::try_from(metadata.slice_start)?;
        let slice_stop = slice_start
            .checked_add(len)
            .ok_or_else(|| vortex_error::vortex_err!("block residual slice length overflows"))?;
        let residual_word_count = usize::try_from(metadata.residual_word_count)?;
        let patch_count = usize::try_from(metadata.patch_count)?;
        let patch_high_count = usize::try_from(metadata.patch_high_count)?;
        vortex_ensure!(buffers.len() == 1, "BlockResidualArray expects one buffer");
        let validity = match children.len() {
            0 => Validity::from(dtype.nullability()),
            1 => Validity::Array(children.get(0, &Validity::DTYPE, unsliced_len)?),
            count => vortex_bail!("BlockResidualArray expects zero or one child, got {count}"),
        };
        let slots = BlockResidualSlots {
            validity: validity_to_child(&validity, unsliced_len),
        }
        .into_slots();
        let data = BlockResidualData::try_new(
            unsliced_len,
            slice_start,
            slice_stop,
            residual_word_count,
            patch_count,
            patch_high_count,
            host_payload(&buffers[0])?,
        )?;
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        BlockResidualSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            decompress_array(array.as_view(), ctx)?.into_array(),
        ))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }
}

impl OperationsVTable<BlockResidual> for BlockResidual {
    fn scalar_at(
        array: ArrayView<'_, BlockResidual>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        if !array.as_ref().is_valid(index, ctx)? {
            return Ok(Scalar::null(array.dtype().clone()));
        }
        let value = scalar_from_array(array, index, ctx)?;
        let nullability = array.dtype().nullability();
        Ok(match array.dtype().as_ptype() {
            PType::U8 => Scalar::primitive(u8::try_from(value)?, nullability),
            PType::U16 => Scalar::primitive(u16::try_from(value)?, nullability),
            PType::U32 => Scalar::primitive(u32::try_from(value)?, nullability),
            PType::U64 => Scalar::primitive(value, nullability),
            PType::I8 => Scalar::primitive(
                i8::from_le_bytes([(u8::try_from(value)? ^ (1_u8 << 7))]),
                nullability,
            ),
            PType::I16 => Scalar::primitive(
                i16::from_le_bytes((u16::try_from(value)? ^ (1_u16 << 15)).to_le_bytes()),
                nullability,
            ),
            PType::I32 => Scalar::primitive(
                i32::from_le_bytes((u32::try_from(value)? ^ (1_u32 << 31)).to_le_bytes()),
                nullability,
            ),
            PType::I64 => Scalar::primitive(
                i64::from_le_bytes((value ^ (1_u64 << 63)).to_le_bytes()),
                nullability,
            ),
            ptype => vortex_bail!("BlockResidual scalar access does not support {ptype}"),
        })
    }
}

impl ValidityVTable<BlockResidual> for BlockResidual {
    fn validity(array: ArrayView<'_, BlockResidual>) -> VortexResult<Validity> {
        array
            .unsliced_validity()
            .slice(array.data().slice_start..array.data().slice_stop)
    }
}

impl SliceReduce for BlockResidual {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let data = array.data().slice(range);
        let parts = ArrayParts::new(BlockResidual, array.dtype().clone(), data.len(), data)
            .with_slots(array.slots().iter().cloned().collect());
        // SAFETY: The source array is valid. The slice only narrows its logical bounds.
        Ok(Some(
            unsafe { Array::from_parts_unchecked(parts) }.into_array(),
        ))
    }
}

static RULES: ParentRuleSet<BlockResidual> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&SliceReduceAdaptor(BlockResidual))]);

pub(crate) trait BlockResidualArrayExt: TypedArrayRef<BlockResidual> {
    fn unsliced_validity(&self) -> Validity {
        child_to_validity(
            self.as_ref().slots()[BlockResidualSlots::VALIDITY].as_ref(),
            self.as_ref().dtype().nullability(),
        )
    }

    /// Return the reference value for each block.
    fn bases(&self) -> &[u64] {
        &self.bases
    }

    /// Return the packed residual width for each block.
    fn residual_widths(&self) -> &[u8] {
        &self.residual_widths
    }

    /// Return the packed patch width for each block.
    fn high_widths(&self) -> &[u8] {
        &self.high_widths
    }

    /// Return the residual payload offsets.
    fn residual_starts(&self) -> &[u32] {
        &self.residual_starts
    }

    /// Return the patch position offsets.
    fn patch_starts(&self) -> &[u32] {
        &self.patch_starts
    }

    /// Return the patch high-bit offsets.
    fn high_starts(&self) -> &[u32] {
        &self.high_starts
    }

    /// Return the packed residual payload.
    fn residual_words(&self) -> &[u64] {
        &self.residual_words
    }

    /// Return the patch positions.
    fn patch_positions(&self) -> &[u16] {
        &self.patch_positions
    }

    /// Return the packed patch high bits.
    fn patch_highs(&self) -> &[u8] {
        &self.patch_highs
    }
}

impl<T: TypedArrayRef<BlockResidual>> BlockResidualArrayExt for T {}

impl BlockResidual {
    /// Estimate the exact encoded size without materializing packed payloads.
    pub fn estimate_primitive(
        array: ArrayView<'_, Primitive>,
    ) -> VortexResult<BlockResidualEstimate> {
        vortex_ensure!(
            array.ptype().is_int(),
            "BlockResidual requires integer values"
        );
        let BlockResidualCodecEstimate {
            encoded_nbytes,
            patch_count,
        } = match array.ptype() {
            PType::U8 => {
                BlockResidualCodec::estimate_transformed(array.as_slice::<u8>(), u64::from)
            }
            PType::U16 => {
                BlockResidualCodec::estimate_transformed(array.as_slice::<u16>(), u64::from)
            }
            PType::U32 => {
                BlockResidualCodec::estimate_transformed(array.as_slice::<u32>(), u64::from)
            }
            PType::U64 => {
                BlockResidualCodec::estimate_transformed(array.as_slice::<u64>(), |value| value)
            }
            PType::I8 => {
                BlockResidualCodec::estimate_transformed(array.as_slice::<i8>(), |value| {
                    u64::from((value as u8) ^ (1_u8 << 7))
                })
            }
            PType::I16 => {
                BlockResidualCodec::estimate_transformed(array.as_slice::<i16>(), |value| {
                    u64::from((value as u16) ^ (1_u16 << 15))
                })
            }
            PType::I32 => {
                BlockResidualCodec::estimate_transformed(array.as_slice::<i32>(), |value| {
                    u64::from((value as u32) ^ (1_u32 << 31))
                })
            }
            PType::I64 => {
                BlockResidualCodec::estimate_transformed(array.as_slice::<i64>(), |value| {
                    (value as u64) ^ (1_u64 << 63)
                })
            }
            ptype => vortex_bail!("BlockResidual does not support {ptype}"),
        };
        let validity_nbytes = validity_to_child(&array.validity()?, array.len())
            .map(|validity| validity.nbytes())
            .unwrap_or(0);
        BlockResidualEstimate::try_new(encoded_nbytes, validity_nbytes, patch_count)
    }

    /// Encode an integer array in independent blocks.
    pub fn from_primitive(array: ArrayView<'_, Primitive>) -> VortexResult<BlockResidualArray> {
        vortex_ensure!(
            array.ptype().is_int(),
            "BlockResidual requires integer values"
        );
        let validity = array.validity()?;
        let values = ordered_values(array)?;
        let parts = BlockResidualCodec::encode_with_word_width(
            &values,
            u8::try_from(array.ptype().bit_width())?,
        )?
        .into_parts()?;
        Self::try_new(parts, validity, array.ptype())
    }

    fn try_new(
        parts: BlockResidualParts,
        validity: Validity,
        ptype: PType,
    ) -> VortexResult<BlockResidualArray> {
        let payload = payload_from_parts(&parts)?;
        let data = BlockResidualData::try_new(
            parts.len,
            0,
            parts.len,
            parts.residual_words.len(),
            parts.patch_positions.len(),
            parts.patch_highs.len(),
            payload,
        )?;
        let slots = BlockResidualSlots {
            validity: validity_to_child(&validity, data.unsliced_len),
        }
        .into_slots();
        Array::try_from_parts(
            ArrayParts::new(
                BlockResidual,
                DType::Primitive(ptype, validity.nullability()),
                data.unsliced_len,
                data,
            )
            .with_slots(slots),
        )
    }
}

fn ordered_values(array: ArrayView<'_, Primitive>) -> VortexResult<Vec<u64>> {
    Ok(match array.ptype() {
        PType::U8 => array
            .as_slice::<u8>()
            .iter()
            .map(|&value| u64::from(value))
            .collect(),
        PType::U16 => array
            .as_slice::<u16>()
            .iter()
            .map(|&value| u64::from(value))
            .collect(),
        PType::U32 => array
            .as_slice::<u32>()
            .iter()
            .map(|&value| u64::from(value))
            .collect(),
        PType::U64 => array.as_slice::<u64>().to_vec(),
        PType::I8 => array
            .as_slice::<i8>()
            .iter()
            .map(|&value| u64::from((value as u8) ^ (1_u8 << 7)))
            .collect(),
        PType::I16 => array
            .as_slice::<i16>()
            .iter()
            .map(|&value| u64::from((value as u16) ^ (1_u16 << 15)))
            .collect(),
        PType::I32 => array
            .as_slice::<i32>()
            .iter()
            .map(|&value| u64::from((value as u32) ^ (1_u32 << 31)))
            .collect(),
        PType::I64 => array
            .as_slice::<i64>()
            .iter()
            .map(|&value| (value as u64) ^ (1_u64 << 63))
            .collect(),
        ptype => vortex_bail!("BlockResidual does not support {ptype}"),
    })
}

impl BlockResidualData {
    fn try_new(
        unsliced_len: usize,
        slice_start: usize,
        slice_stop: usize,
        residual_word_count: usize,
        patch_count: usize,
        patch_high_count: usize,
        payload: ByteBuffer,
    ) -> VortexResult<Self> {
        let block_count = unsliced_len.div_ceil(BLOCK_LEN);
        let mut offset = 0;
        let bases = take_payload(&payload, &mut offset, block_count, "bases")?;
        let residual_words =
            take_payload(&payload, &mut offset, residual_word_count, "residual words")?;
        let residual_starts =
            take_payload(&payload, &mut offset, block_count + 1, "residual starts")?;
        let patch_starts = take_payload(&payload, &mut offset, block_count + 1, "patch starts")?;
        let high_starts = take_payload(&payload, &mut offset, block_count + 1, "high starts")?;
        let patch_positions = take_payload(&payload, &mut offset, patch_count, "patch positions")?;
        let residual_widths = take_payload(&payload, &mut offset, block_count, "residual widths")?;
        let high_widths = take_payload(&payload, &mut offset, block_count, "high widths")?;
        let patch_highs = take_payload(&payload, &mut offset, patch_high_count, "patch highs")?;
        vortex_ensure!(
            offset == payload.len(),
            "block residual payload contains trailing bytes"
        );
        Ok(Self {
            unsliced_len,
            slice_start,
            slice_stop,
            payload,
            bases,
            residual_widths,
            high_widths,
            residual_starts,
            patch_starts,
            high_starts,
            residual_words,
            patch_positions,
            patch_highs,
        })
    }

    fn validate(
        &self,
        dtype: &DType,
        len: usize,
        _slots: BlockResidualSlotsView<'_>,
        validity: &Validity,
    ) -> VortexResult<()> {
        vortex_ensure!(
            dtype.is_int(),
            "BlockResidualArray requires an integer dtype"
        );
        vortex_ensure!(
            self.slice_start <= self.slice_stop && self.slice_stop <= self.unsliced_len,
            "block residual slice exceeds its source length"
        );
        vortex_ensure!(len == self.len(), "block residual slice length is invalid");
        let block_count = self.unsliced_len.div_ceil(BLOCK_LEN);
        vortex_ensure!(
            self.bases.len() == block_count
                && self.residual_widths.len() == block_count
                && self.high_widths.len() == block_count,
            "block residual block tables have invalid lengths"
        );
        vortex_ensure!(
            self.residual_starts.len() == block_count + 1
                && self.patch_starts.len() == block_count + 1
                && self.high_starts.len() == block_count + 1,
            "block residual offset tables have invalid lengths"
        );
        validate_offset_table(
            &self.residual_starts,
            block_count,
            self.residual_words.len(),
            "residual",
        )?;
        validate_offset_table(
            &self.patch_starts,
            block_count,
            self.patch_positions.len(),
            "patch",
        )?;
        validate_offset_table(
            &self.high_starts,
            block_count,
            self.patch_highs.len(),
            "patch high",
        )?;
        let logical_width = dtype.as_ptype().bit_width();
        let maximum = if logical_width == 64 {
            u64::MAX
        } else {
            (1_u64 << logical_width) - 1
        };
        for block_index in 0..block_count {
            vortex_ensure!(
                self.bases[block_index] <= maximum,
                "block residual base exceeds its logical type"
            );
            let residual_width = self.residual_widths[block_index];
            let high_width = self.high_widths[block_index];
            vortex_ensure!(
                usize::from(residual_width) <= logical_width
                    && usize::from(high_width) <= logical_width
                    && usize::from(residual_width) + usize::from(high_width) <= logical_width,
                "block residual bit widths are invalid"
            );
            let residual_range = payload_range(
                &self.residual_starts,
                block_index,
                self.residual_words.len(),
                "residual",
            )?;
            vortex_ensure!(
                residual_range.len() == BLOCK_LEN * usize::from(residual_width) / 64,
                "block residual word count is invalid"
            );
            let patch_range = payload_range(
                &self.patch_starts,
                block_index,
                self.patch_positions.len(),
                "patch",
            )?;
            let high_range = payload_range(
                &self.high_starts,
                block_index,
                self.patch_highs.len(),
                "patch high",
            )?;
            let positions = &self.patch_positions[patch_range];
            validate_patch_header(
                residual_width,
                high_width,
                positions.len(),
                high_range.len(),
            )?;
            let block_start = block_index * BLOCK_LEN;
            let block_len = (self.unsliced_len - block_start).min(BLOCK_LEN);
            let mut previous = None;
            for &position in positions {
                validate_patch_position(block_len, previous, position)?;
                previous = Some(position);
            }
        }
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == self.unsliced_len,
                "block residual validity length is invalid"
            );
        }
        Ok(())
    }

    fn len(&self) -> usize {
        self.slice_stop - self.slice_start
    }

    fn slice(&self, range: Range<usize>) -> Self {
        Self {
            slice_start: self.slice_start + range.start,
            slice_stop: self.slice_start + range.end,
            ..self.clone()
        }
    }

    fn replace_payload(&mut self, buffer: &BufferHandle) -> VortexResult<()> {
        *self = Self::try_new(
            self.unsliced_len,
            self.slice_start,
            self.slice_stop,
            self.residual_words.len(),
            self.patch_positions.len(),
            self.patch_highs.len(),
            host_payload(buffer)?,
        )?;
        Ok(())
    }
}

fn host_payload(buffer: &BufferHandle) -> VortexResult<ByteBuffer> {
    buffer
        .clone()
        .ensure_aligned(Alignment::of::<u64>())?
        .try_into_host_sync()
}

fn take_payload<T: NativePType>(
    payload: &ByteBuffer,
    offset: &mut usize,
    len: usize,
    name: &str,
) -> VortexResult<Buffer<T>> {
    let nbytes = len
        .checked_mul(size_of::<T>())
        .ok_or_else(|| vortex_error::vortex_err!("block residual {name} size overflows"))?;
    let stop = offset
        .checked_add(nbytes)
        .ok_or_else(|| vortex_error::vortex_err!("block residual {name} offset overflows"))?;
    vortex_ensure!(
        stop <= payload.len(),
        "block residual {name} exceeds the payload"
    );
    let bytes = payload.slice_with_alignment(*offset..stop, Alignment::of::<T>());
    *offset = stop;
    Ok(Buffer::from_byte_buffer(bytes))
}

fn payload_from_parts(parts: &BlockResidualParts) -> VortexResult<ByteBuffer> {
    let total_nbytes = [
        size_of_val(parts.bases.as_slice()),
        size_of_val(parts.residual_words.as_slice()),
        size_of_val(parts.residual_starts.as_slice()),
        size_of_val(parts.patch_starts.as_slice()),
        size_of_val(parts.high_starts.as_slice()),
        size_of_val(parts.patch_positions.as_slice()),
        size_of_val(parts.residual_widths.as_slice()),
        size_of_val(parts.high_widths.as_slice()),
        size_of_val(parts.patch_highs.as_slice()),
    ]
    .into_iter()
    .try_fold(0_usize, |total, nbytes| total.checked_add(nbytes))
    .ok_or_else(|| vortex_error::vortex_err!("block residual payload size overflows"))?;
    let mut payload = ByteBufferMut::with_capacity_aligned(total_nbytes, Alignment::of::<u64>());
    append_native(&mut payload, &parts.bases);
    append_native(&mut payload, &parts.residual_words);
    append_native(&mut payload, &parts.residual_starts);
    append_native(&mut payload, &parts.patch_starts);
    append_native(&mut payload, &parts.high_starts);
    append_native(&mut payload, &parts.patch_positions);
    append_native(&mut payload, &parts.residual_widths);
    append_native(&mut payload, &parts.high_widths);
    append_native(&mut payload, &parts.patch_highs);
    Ok(payload.freeze())
}

fn append_native<T: NativePType>(payload: &mut ByteBufferMut, values: &[T]) {
    // SAFETY: NativePType values contain no padding and permit every initialized bit pattern.
    let bytes =
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), size_of_val(values)) };
    payload.extend_from_slice(bytes);
}

fn decode_array_values<T: NativePType, U: NativePType + ResidualWord, const DIRECT_OUTPUT: bool>(
    array: ArrayView<'_, BlockResidual>,
    _ctx: &mut ExecutionCtx,
    mut transform: impl FnMut(U) -> T,
) -> VortexResult<PrimitiveArray> {
    let bases = array.bases();
    let residual_widths = array.residual_widths();
    let high_widths = array.high_widths();
    let residual_starts = array.residual_starts();
    let patch_starts = array.patch_starts();
    let high_starts = array.high_starts();
    let residual_words = array.residual_words();
    let patch_positions = array.patch_positions();
    let patch_highs = array.patch_highs();
    let logical_range = array.data().slice_start..array.data().slice_stop;
    let mut direct_values =
        DIRECT_OUTPUT.then(|| BufferMut::<U>::with_capacity(logical_range.len()));
    let mut transformed_values =
        (!DIRECT_OUTPUT).then(|| BufferMut::<T>::with_capacity(logical_range.len()));
    let mut residuals = [U::default(); BLOCK_LEN];

    let first_block = logical_range.start / BLOCK_LEN;
    let last_block = logical_range.end.div_ceil(BLOCK_LEN);
    for block_index in first_block..last_block {
        let block_start = block_index * BLOCK_LEN;
        let block_len = (array.data().unsliced_len - block_start).min(BLOCK_LEN);
        let block_stop = block_start + block_len;
        if block_stop <= logical_range.start || block_start >= logical_range.end {
            continue;
        }

        let residual_width = residual_widths[block_index];
        let high_width = high_widths[block_index];
        let base = <U as ResidualWord>::from_u64(bases[block_index]);
        vortex_ensure!(
            residual_width <= U::BITS
                && high_width <= U::BITS
                && u16::from(residual_width) + u16::from(high_width) <= u16::from(U::BITS),
            "block residual bit widths are invalid"
        );
        let residual_payload = payload_range(
            residual_starts,
            block_index,
            residual_words.len(),
            "residual",
        )?;
        vortex_ensure!(
            residual_payload.len() == BLOCK_LEN * usize::from(residual_width) / 64,
            "block residual word count is invalid"
        );
        let patch_payload =
            payload_range(patch_starts, block_index, patch_positions.len(), "patch")?;
        let high_payload =
            payload_range(high_starts, block_index, patch_highs.len(), "patch high")?;
        let positions = &patch_positions[patch_payload];
        validate_patch_header(
            residual_width,
            high_width,
            positions.len(),
            high_payload.len(),
        )?;
        let highs = &patch_highs[high_payload];
        let local_start = logical_range.start.saturating_sub(block_start);
        let local_stop = (logical_range.end - block_start).min(block_len);

        if residual_width == 0 {
            let output_start = decoded_len(direct_values.as_ref(), transformed_values.as_ref());
            append_repeated_decoded::<T, U, DIRECT_OUTPUT>(
                &mut direct_values,
                &mut transformed_values,
                base,
                local_stop - local_start,
                &mut transform,
            );
            let mut previous_position = None;
            for (patch_index, &position) in positions.iter().enumerate() {
                validate_patch_position(block_len, previous_position, position)?;
                previous_position = Some(position);
                let position = usize::from(position);
                if position < local_start || position >= local_stop {
                    continue;
                }
                // SAFETY: The payload includes fifteen readable padding bytes.
                let high = unsafe {
                    read_wide_bits(highs, patch_index * usize::from(high_width), high_width)
                };
                set_decoded::<T, U, DIRECT_OUTPUT>(
                    &mut direct_values,
                    &mut transformed_values,
                    output_start + position - local_start,
                    base.wrapping_add(<U as ResidualWord>::from_u64(high)),
                    &mut transform,
                );
            }
            continue;
        }

        let packed = packed_words_as_native::<U>(&residual_words[residual_payload]);
        if positions.is_empty() && DIRECT_OUTPUT && local_start == 0 && local_stop == BLOCK_LEN {
            // SAFETY: The encoder writes one complete FastLanes chunk, and the output has capacity.
            unsafe {
                append_unpacked_add(
                    direct_values
                        .as_mut()
                        .vortex_expect("direct BlockResidual output is present"),
                    usize::from(residual_width),
                    packed,
                    base,
                );
            }
            continue;
        }
        // SAFETY: The encoder writes one complete FastLanes chunk for each block.
        unsafe {
            if positions.is_empty() && DIRECT_OUTPUT {
                U::unpack_add(usize::from(residual_width), packed, base, &mut residuals);
            } else {
                U::unchecked_unpack(usize::from(residual_width), packed, &mut residuals);
            }
        }
        if positions.is_empty() && DIRECT_OUTPUT {
            append_decoded::<T, U, DIRECT_OUTPUT>(
                &mut direct_values,
                &mut transformed_values,
                &residuals[local_start..local_stop],
                &mut transform,
            );
            continue;
        }
        let mut previous_position = None;
        for (patch_index, &position) in positions.iter().enumerate() {
            validate_patch_position(block_len, previous_position, position)?;
            previous_position = Some(position);
            // SAFETY: The payload includes fifteen readable padding bytes.
            let high =
                unsafe { read_wide_bits(highs, patch_index * usize::from(high_width), high_width) };
            residuals[usize::from(position)].apply_high(high, residual_width);
        }

        append_residuals::<T, U, DIRECT_OUTPUT>(
            &mut direct_values,
            &mut transformed_values,
            &mut residuals[local_start..local_stop],
            base,
            &mut transform,
        );
    }
    let validity = array.validity()?;
    if DIRECT_OUTPUT {
        Ok(PrimitiveArray::new(
            direct_values
                .vortex_expect("direct BlockResidual output is present")
                .freeze(),
            validity,
        ))
    } else {
        Ok(PrimitiveArray::new(
            transformed_values
                .vortex_expect("transformed BlockResidual output is present")
                .freeze(),
            validity,
        ))
    }
}

fn decoded_len<T: NativePType, U: NativePType>(
    direct: Option<&BufferMut<U>>,
    transformed: Option<&BufferMut<T>>,
) -> usize {
    direct.map_or_else(|| transformed.map_or(0, BufferMut::len), BufferMut::len)
}

fn append_repeated_decoded<T: NativePType, U: NativePType, const DIRECT_OUTPUT: bool>(
    direct: &mut Option<BufferMut<U>>,
    transformed: &mut Option<BufferMut<T>>,
    value: U,
    count: usize,
    transform: &mut impl FnMut(U) -> T,
) {
    if DIRECT_OUTPUT {
        append_repeated(
            direct
                .as_mut()
                .vortex_expect("direct BlockResidual output is present"),
            value,
            count,
        );
    } else {
        let output = transformed
            .as_mut()
            .vortex_expect("transformed BlockResidual output is present");
        let output_len = output.len();
        for destination in &mut output.spare_capacity_mut()[..count] {
            destination.write(transform(value));
        }
        // SAFETY: The loop initialized each new output value.
        unsafe { output.set_len(output_len + count) };
    }
}

fn append_decoded<T: NativePType, U: NativePType, const DIRECT_OUTPUT: bool>(
    direct: &mut Option<BufferMut<U>>,
    transformed: &mut Option<BufferMut<T>>,
    values: &[U],
    transform: &mut impl FnMut(U) -> T,
) {
    if DIRECT_OUTPUT {
        append_values(
            direct
                .as_mut()
                .vortex_expect("direct BlockResidual output is present"),
            values,
        );
    } else {
        let output = transformed
            .as_mut()
            .vortex_expect("transformed BlockResidual output is present");
        let output_len = output.len();
        for (destination, &value) in output.spare_capacity_mut()[..values.len()]
            .iter_mut()
            .zip(values)
        {
            destination.write(transform(value));
        }
        // SAFETY: The loop initialized each new output value.
        unsafe { output.set_len(output_len + values.len()) };
    }
}

fn append_residuals<T: NativePType, U: NativePType + ResidualWord, const DIRECT_OUTPUT: bool>(
    direct: &mut Option<BufferMut<U>>,
    transformed: &mut Option<BufferMut<T>>,
    residuals: &mut [U],
    base: U,
    transform: &mut impl FnMut(U) -> T,
) {
    if DIRECT_OUTPUT {
        for residual in residuals.iter_mut() {
            *residual = residual.wrapping_add(base);
        }
        append_values(
            direct
                .as_mut()
                .vortex_expect("direct BlockResidual output is present"),
            residuals,
        );
    } else {
        let output = transformed
            .as_mut()
            .vortex_expect("transformed BlockResidual output is present");
        let output_len = output.len();
        for (destination, &residual) in output.spare_capacity_mut()[..residuals.len()]
            .iter_mut()
            .zip(residuals.iter())
        {
            destination.write(transform(residual.wrapping_add(base)));
        }
        // SAFETY: The loop initialized each new output value.
        unsafe { output.set_len(output_len + residuals.len()) };
    }
}

fn set_decoded<T: NativePType, U: NativePType, const DIRECT_OUTPUT: bool>(
    direct: &mut Option<BufferMut<U>>,
    transformed: &mut Option<BufferMut<T>>,
    index: usize,
    value: U,
    transform: &mut impl FnMut(U) -> T,
) {
    if DIRECT_OUTPUT {
        direct
            .as_mut()
            .vortex_expect("direct BlockResidual output is present")[index] = value;
    } else {
        transformed
            .as_mut()
            .vortex_expect("transformed BlockResidual output is present")[index] = transform(value);
    }
}

fn append_repeated<T: NativePType>(output: &mut BufferMut<T>, value: T, count: usize) {
    let output_len = output.len();
    for destination in &mut output.spare_capacity_mut()[..count] {
        destination.write(value);
    }
    // SAFETY: The loop initialized each new output value.
    unsafe { output.set_len(output_len + count) };
}

fn append_values<T: NativePType>(output: &mut BufferMut<T>, values: &[T]) {
    let output_len = output.len();
    for (destination, &value) in output.spare_capacity_mut()[..values.len()]
        .iter_mut()
        .zip(values)
    {
        destination.write(value);
    }
    // SAFETY: The loop initialized each new output value.
    unsafe { output.set_len(output_len + values.len()) };
}

unsafe fn append_unpacked_add<T: NativePType + ResidualWord>(
    output: &mut BufferMut<T>,
    bit_width: usize,
    packed: &[T],
    base: T,
) {
    let output_len = output.len();
    let destination = output.spare_capacity_mut()[..BLOCK_LEN].as_mut_ptr();
    // SAFETY: The caller guarantees capacity. FastLanes initializes one complete output chunk.
    let destination = unsafe { std::slice::from_raw_parts_mut(destination.cast::<T>(), BLOCK_LEN) };
    // SAFETY: The caller provides one complete FastLanes input and output chunk.
    unsafe { T::unpack_add(bit_width, packed, base, destination) };
    // SAFETY: FastLanes initialized each new output value.
    unsafe { output.set_len(output_len + BLOCK_LEN) };
}

fn decompress_array(
    array: ArrayView<'_, BlockResidual>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    match array.dtype().as_ptype() {
        PType::U8 => decode_array_values::<u8, u8, true>(array, ctx, |value| value),
        PType::U16 => decode_array_values::<u16, u16, true>(array, ctx, |value| value),
        PType::U32 => decode_array_values::<u32, u32, true>(array, ctx, |value| value),
        PType::U64 => decode_array_values::<u64, u64, true>(array, ctx, |value| value),
        PType::I8 => {
            decode_array_values::<i8, u8, false>(array, ctx, |value| (value ^ (1_u8 << 7)) as i8)
        }
        PType::I16 => decode_array_values::<i16, u16, false>(array, ctx, |value| {
            (value ^ (1_u16 << 15)) as i16
        }),
        PType::I32 => decode_array_values::<i32, u32, false>(array, ctx, |value| {
            (value ^ (1_u32 << 31)) as i32
        }),
        PType::I64 => decode_array_values::<i64, u64, false>(array, ctx, |value| {
            (value ^ (1_u64 << 63)) as i64
        }),
        ptype => vortex_bail!("BlockResidual decode does not support {ptype}"),
    }
}

pub(crate) fn decompress_ordered_f32(
    array: ArrayView<'_, BlockResidual>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    decode_array_values::<f32, u32, false>(array, ctx, |ordered| {
        let bits = if ordered & (1_u32 << 31) == 0 {
            !ordered
        } else {
            ordered ^ (1_u32 << 31)
        };
        f32::from_bits(bits)
    })
}

pub(crate) fn decompress_ordered_f16(
    array: ArrayView<'_, BlockResidual>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    decode_array_values::<f16, u16, false>(array, ctx, |ordered| {
        let bits = if ordered & (1_u16 << 15) == 0 {
            !ordered
        } else {
            ordered ^ (1_u16 << 15)
        };
        f16::from_bits(bits)
    })
}

pub(crate) fn decompress_ordered_f64(
    array: ArrayView<'_, BlockResidual>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    decode_array_values::<f64, u64, false>(array, ctx, |ordered| {
        let bits = if ordered & (1_u64 << 63) == 0 {
            !ordered
        } else {
            ordered ^ (1_u64 << 63)
        };
        f64::from_bits(bits)
    })
}

fn scalar_from_array(
    array: ArrayView<'_, BlockResidual>,
    index: usize,
    _ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    let source_index = array.data().slice_start + index;
    let block_index = source_index / BLOCK_LEN;
    let index_in_block = source_index % BLOCK_LEN;
    let residual_width = array.residual_widths()[block_index];
    let high_width = array.high_widths()[block_index];
    let logical_width = array.dtype().as_ptype().bit_width();
    vortex_ensure!(
        usize::from(residual_width) <= logical_width
            && usize::from(high_width) <= logical_width
            && usize::from(residual_width) + usize::from(high_width) <= logical_width,
        "block residual bit widths are invalid"
    );
    let residual_words = array.residual_words();
    let residual_payload = payload_range(
        array.residual_starts(),
        block_index,
        residual_words.len(),
        "residual",
    )?;
    vortex_ensure!(
        residual_payload.len() == BLOCK_LEN * usize::from(residual_width) / 64,
        "block residual word count is invalid"
    );
    let mut residual = match logical_width {
        8 => unpack_single_residual::<u8>(
            residual_width,
            &residual_words[residual_payload],
            index_in_block,
        ),
        16 => unpack_single_residual::<u16>(
            residual_width,
            &residual_words[residual_payload],
            index_in_block,
        ),
        32 => unpack_single_residual::<u32>(
            residual_width,
            &residual_words[residual_payload],
            index_in_block,
        ),
        64 => unpack_single_residual::<u64>(
            residual_width,
            &residual_words[residual_payload],
            index_in_block,
        ),
        _ => vortex_bail!("block residual logical bit width is invalid"),
    };

    let positions = array.patch_positions();
    let patch_payload = payload_range(array.patch_starts(), block_index, positions.len(), "patch")?;
    let block_positions = &positions[patch_payload];
    let highs = array.patch_highs();
    let high_payload = payload_range(array.high_starts(), block_index, highs.len(), "patch high")?;
    validate_patch_header(
        residual_width,
        high_width,
        block_positions.len(),
        high_payload.len(),
    )?;
    if let Ok(patch_index) = block_positions.binary_search(&u16::try_from(index_in_block)?) {
        let high_payload = &highs[high_payload];
        // SAFETY: The payload includes fifteen readable padding bytes.
        let high = unsafe {
            read_wide_bits(
                high_payload,
                patch_index * usize::from(high_width),
                high_width,
            )
        };
        residual |= high << residual_width;
    }
    Ok(array.bases()[block_index].wrapping_add(residual))
}

fn unpack_single_residual<T: ResidualWord>(width: u8, packed_words: &[u64], index: usize) -> u64 {
    if width == 0 {
        return 0;
    }
    let packed = packed_words_as_native::<T>(packed_words);
    // SAFETY: The encoder writes one complete FastLanes chunk for each block.
    unsafe { T::unchecked_unpack_single(usize::from(width), packed, index).to_u64() }
}

fn validate_patch_header(
    residual_width: u8,
    high_width: u8,
    patch_count: usize,
    high_payload_len: usize,
) -> VortexResult<()> {
    vortex_ensure!(
        patch_count == 0 || (high_width > 0 && residual_width < 64),
        "block residual patches require nonzero high bits"
    );
    let expected_high_len = if patch_count == 0 {
        0
    } else {
        (patch_count * usize::from(high_width)).div_ceil(8) + 15
    };
    vortex_ensure!(
        high_payload_len == expected_high_len,
        "block residual patch high payload is invalid"
    );
    Ok(())
}

#[inline(always)]
fn validate_patch_position(
    block_len: usize,
    previous_position: Option<u16>,
    position: u16,
) -> VortexResult<()> {
    if usize::from(position) < block_len
        && previous_position.is_none_or(|previous| previous < position)
    {
        Ok(())
    } else {
        invalid_patch_position()
    }
}

#[cold]
#[inline(never)]
fn invalid_patch_position() -> VortexResult<()> {
    vortex_bail!("block residual patch positions are invalid")
}

fn validate_offset_table(
    starts: &[u32],
    block_count: usize,
    payload_len: usize,
    name: &str,
) -> VortexResult<()> {
    vortex_ensure!(
        starts.len() == block_count + 1,
        "block residual {name} offsets have an invalid length"
    );
    vortex_ensure!(
        starts.first() == Some(&0)
            && starts.last().copied().map(usize::try_from).transpose()? == Some(payload_len),
        "block residual {name} offsets do not cover the payload"
    );
    vortex_ensure!(
        starts.windows(2).all(|window| window[0] <= window[1]),
        "block residual {name} offsets are not ordered"
    );
    Ok(())
}

fn payload_range(
    starts: &[u32],
    block_index: usize,
    payload_len: usize,
    name: &str,
) -> VortexResult<Range<usize>> {
    let start = usize::try_from(
        *starts
            .get(block_index)
            .ok_or_else(|| vortex_error::vortex_err!("block residual {name} start is missing"))?,
    )?;
    let stop = usize::try_from(
        *starts
            .get(block_index + 1)
            .ok_or_else(|| vortex_error::vortex_err!("block residual {name} stop is missing"))?,
    )?;
    vortex_ensure!(
        start <= stop && stop <= payload_len,
        "block residual {name} offsets are invalid"
    );
    Ok(start..stop)
}

#[derive(Clone, Copy)]
struct BlockResidualMetadata {
    unsliced_len: u64,
    slice_start: u64,
    residual_word_count: u64,
    patch_count: u64,
    patch_high_count: u64,
}

impl BlockResidualMetadata {
    fn from_data(data: &BlockResidualData) -> VortexResult<Self> {
        Ok(Self {
            unsliced_len: u64::try_from(data.unsliced_len)?,
            slice_start: u64::try_from(data.slice_start)?,
            residual_word_count: u64::try_from(data.residual_words.len())?,
            patch_count: u64::try_from(data.patch_positions.len())?,
            patch_high_count: u64::try_from(data.patch_highs.len())?,
        })
    }

    fn encode(self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(METADATA_LEN);
        bytes.push(METADATA_VERSION);
        for value in [
            self.unsliced_len,
            self.slice_start,
            self.residual_word_count,
            self.patch_count,
            self.patch_high_count,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn decode(bytes: &[u8]) -> VortexResult<Self> {
        vortex_ensure!(
            bytes.len() == METADATA_LEN,
            "BlockResidualArray metadata requires {METADATA_LEN} bytes"
        );
        vortex_ensure!(
            bytes[0] == METADATA_VERSION,
            "unsupported BlockResidualArray metadata version {}",
            bytes[0]
        );
        let read = |offset: usize| {
            u64::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
            ])
        };
        Ok(Self {
            unsliced_len: read(1),
            slice_start: read(9),
            residual_word_count: read(17),
            patch_count: read(25),
            patch_high_count: read(33),
        })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::dtype::NativePType;
    use vortex_array::dtype::PType;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_error::VortexResult;
    use vortex_session::registry::ReadContext;

    use super::BlockResidual;
    use super::BlockResidualArrayExt;
    use crate::BlockResidualCodec;

    #[test]
    fn roundtrip_and_scalar_access() -> VortexResult<()> {
        let values = (0..4_099)
            .map(|index| Ok(1_000_000_u64 + u64::try_from(index * index)?))
            .collect::<VortexResult<Vec<_>>>()?;
        let primitive = PrimitiveArray::from_iter(values.clone());
        let encoded = BlockResidual::from_primitive(primitive.as_view())?;
        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        assert_arrays_eq!(encoded.clone(), primitive.into_array(), &mut ctx);
        for index in [0, 1, 1_023, 1_024, 4_098] {
            let scalar = encoded.execute_scalar(index, &mut ctx)?;
            assert_eq!(
                scalar.as_primitive().typed_value::<u64>(),
                Some(values[index])
            );
        }
        Ok(())
    }

    #[test]
    fn signed_roundtrip_and_scalar_access() -> VortexResult<()> {
        let values = (0..2_050)
            .map(|index| match index {
                0 => i64::MIN,
                1_023 => -1,
                1_024 => 0,
                2_049 => i64::MAX,
                _ => (index as i64 - 1_025) * 1_000_003,
            })
            .collect::<Vec<_>>();
        let primitive = PrimitiveArray::from_iter(values.clone());
        let encoded = BlockResidual::from_primitive(primitive.as_view())?;
        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();

        assert_arrays_eq!(encoded.clone(), primitive.into_array(), &mut ctx);
        for index in [0, 1, 1_023, 1_024, 2_049] {
            let scalar = encoded.execute_scalar(index, &mut ctx)?;
            assert_eq!(
                scalar.as_primitive().typed_value::<i64>(),
                Some(values[index])
            );
        }
        Ok(())
    }

    #[rstest]
    #[case(vec![0_u8, 1, u8::MAX])]
    #[case(vec![0_u16, 1, u16::MAX])]
    #[case(vec![0_u32, 1, u32::MAX])]
    #[case(vec![0_u64, 1, u64::MAX])]
    #[case(vec![i8::MIN, -1, 0, 1, i8::MAX])]
    #[case(vec![i16::MIN, -1, 0, 1, i16::MAX])]
    #[case(vec![i32::MIN, -1, 0, 1, i32::MAX])]
    #[case(vec![i64::MIN, -1, 0, 1, i64::MAX])]
    fn integer_ptype_roundtrip<T>(#[case] values: Vec<T>) -> VortexResult<()>
    where
        T: NativePType + Copy,
    {
        let primitive = PrimitiveArray::from_iter(values);
        let encoded = BlockResidual::from_primitive(primitive.as_view())?;
        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();

        assert_arrays_eq!(encoded, primitive, &mut ctx);
        Ok(())
    }

    #[test]
    fn rejects_components_outside_logical_width() -> VortexResult<()> {
        let mut parts =
            BlockResidualCodec::encode_with_word_width(&[0_u64, 1, 2], 64)?.into_parts()?;
        parts.bases[0] = u64::from(u8::MAX) + 1;
        assert!(BlockResidual::try_new(parts, Validity::NonNullable, PType::U8).is_err());
        Ok(())
    }

    #[test]
    fn rejects_invalid_component_offsets() -> VortexResult<()> {
        let mut parts =
            BlockResidualCodec::encode_with_word_width(&[0_u64, 1, 2], 64)?.into_parts()?;
        parts.residual_starts[0] = 1;
        assert!(BlockResidual::try_new(parts, Validity::NonNullable, PType::U64).is_err());
        Ok(())
    }

    #[test]
    fn rejects_non_integer_input() {
        let primitive = PrimitiveArray::from_iter([0.0_f32, 1.0, 2.0]);
        assert!(BlockResidual::from_primitive(primitive.as_view()).is_err());
    }

    #[test]
    fn u32_direct_decode_roundtrip() -> VortexResult<()> {
        let primitive = PrimitiveArray::from_iter((0..2_050_u32).map(|index| {
            let block = index / 1_024;
            block * 1_000_000 + (index * 7_919) % 1_024
        }));
        let encoded = BlockResidual::from_primitive(primitive.as_view())?;
        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();

        assert_arrays_eq!(encoded, primitive.clone().into_array(), &mut ctx);
        assert_arrays_eq!(
            encoded.into_array().slice(1_023..1_026)?,
            primitive.into_array().slice(1_023..1_026)?,
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn nullable_slice_and_scalar_access() -> VortexResult<()> {
        let values = (0..2_050)
            .map(|index| Ok(u64::try_from(index * index)?))
            .collect::<VortexResult<Vec<_>>>()?;
        let validity = Validity::from_iter((0..values.len()).map(|index| index != 1_024));
        let primitive = PrimitiveArray::new(Buffer::from(values), validity);
        let encoded = BlockResidual::from_primitive(primitive.as_view())?;
        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();

        assert!(encoded.execute_scalar(1_024, &mut ctx)?.is_null());
        let sliced = encoded.into_array().slice(1_023..1_026)?;
        let expected = primitive.into_array().slice(1_023..1_026)?;
        assert_arrays_eq!(sliced, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn zero_width_patched_block_roundtrip() -> VortexResult<()> {
        let mut values = vec![42_u32; 2_050];
        values[1_023] = u32::MAX;
        let validity = Validity::from_iter((0..values.len()).map(|index| index != 1_024));
        let primitive = PrimitiveArray::new(Buffer::from(values.clone()), validity);
        let encoded = BlockResidual::from_primitive(primitive.as_view())?;
        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        assert_eq!(encoded.residual_widths()[0], 0);
        assert_eq!(
            encoded
                .execute_scalar(1_023, &mut ctx)?
                .as_primitive()
                .typed_value::<u32>(),
            Some(u32::MAX)
        );
        assert!(encoded.execute_scalar(1_024, &mut ctx)?.is_null());
        assert_arrays_eq!(
            encoded.into_array().slice(1_022..1_025)?,
            primitive.into_array().slice(1_022..1_025)?,
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn estimate_matches_materialized_size() -> VortexResult<()> {
        let mut values = vec![42_u32; 2_050];
        values[1_023] = u32::MAX;
        values[2_049] = u32::MAX - 1;
        let validity = Validity::from_iter((0..values.len()).map(|index| index != 1_024));
        let primitive = PrimitiveArray::new(Buffer::from(values), validity);
        let estimate = BlockResidual::estimate_primitive(primitive.as_view())?;
        let encoded = BlockResidual::from_primitive(primitive.as_view())?;

        assert_eq!(estimate.nbytes(), encoded.nbytes());
        assert_eq!(estimate.patch_count(), encoded.patch_positions().len());
        Ok(())
    }

    #[test]
    fn nullable_slice_serialization_roundtrip() -> VortexResult<()> {
        let values = (0..2_050)
            .map(|index| Ok(u64::try_from(index * index)?))
            .collect::<VortexResult<Vec<_>>>()?;
        let validity = Validity::from_iter((0..values.len()).map(|index| index != 1_024));
        let primitive = PrimitiveArray::new(Buffer::from(values), validity);
        let sliced = BlockResidual::from_primitive(primitive.as_view())?
            .into_array()
            .slice(1_023..1_026)?;
        let expected = primitive.into_array().slice(1_023..1_026)?;
        let dtype = sliced.dtype().clone();
        let len = sliced.len();
        let array_context = ArrayContext::empty();
        let session = array_session();
        crate::initialize(&session);
        let serialized =
            sliced.serialize(&array_context, &session, &SerializeOptions::default())?;
        let mut bytes = ByteBufferMut::empty();
        for buffer in serialized {
            bytes.extend_from_slice(buffer.as_ref());
        }
        let decoded = SerializedArray::try_from(bytes.freeze())?.decode(
            &dtype,
            len,
            &ReadContext::new(array_context.to_ids()),
            &session,
        )?;
        assert!(decoded.is::<BlockResidual>());
        assert_arrays_eq!(decoded, expected, &mut session.create_execution_ctx());
        Ok(())
    }

    #[test]
    fn conformance() -> VortexResult<()> {
        let mut values = vec![42_i16; 2_050];
        values[1_023] = i16::MAX;
        let primitive = PrimitiveArray::new(
            Buffer::from(values),
            Validity::from_iter((0..2_050).map(|index| index != 1_024)),
        );
        let array = BlockResidual::from_primitive(primitive.as_view())?.into_array();
        let session = array_session();
        crate::initialize(&session);
        test_array_consistency(&array, &mut session.create_execution_ctx());
        Ok(())
    }
}
