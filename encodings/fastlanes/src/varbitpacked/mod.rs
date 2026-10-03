// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bit-packing with one bit width per 1024-element chunk.
//!
//! [`BitPacked`](crate::BitPacked) packs a whole array at one width and patches the values that do
//! not fit. When value ranges drift across an array, as residuals of per-chunk models often do, one
//! wide chunk forces that width onto every chunk or turns into many patches. [`VarBitPacked`] packs
//! each chunk at the width of its own largest value instead, with no patches, as LeCo and Parquet's
//! miniblocks do.

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::mem::MaybeUninit;
use std::ops::Range;
use std::sync::Arc;

use fastlanes::BitPacking;
use num_traits::AsPrimitive;
use num_traits::PrimInt;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArraySlots;
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::TypedArrayRef;
use vortex_array::array_slots;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::require_validity;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::FL_CHUNK_SIZE;

mod plugin;
#[cfg(test)]
mod tests;

pub use plugin::VarBitPackedPlugin;

/// The serialized and in-memory ID of [`VarBitPacked`].
pub fn var_bitpacked_id() -> ArrayId {
    static ID: CachedId = CachedId::new("fastlanes.varbitpacked");
    *ID
}

#[array_slots(VarBitPacked)]
pub struct VarBitPackedSlots {
    /// The validity bitmap indicating which elements are non-null.
    #[slot(0)]
    pub validity_child: Option<ArrayRef>,
}

/// A [`VarBitPacked`]-encoded array.
pub type VarBitPackedArray = Array<VarBitPacked>;

#[derive(Clone, Debug)]
pub struct VarBitPacked;

/// The packed chunks, their widths, and where each chunk starts.
#[derive(Clone, Debug)]
pub struct VarBitPackedData {
    /// The position of the first element within the first chunk.
    pub(crate) offset: u16,
    /// Every chunk's packed words, concatenated, in the array's unsigned physical type.
    pub(crate) packed: BufferHandle,
    /// One bit width per chunk, as `u8`.
    pub(crate) widths: BufferHandle,
    /// The index in `packed`, in words, where each chunk starts, plus the total. Derived from
    /// `widths`, not serialized.
    pub(crate) starts: Arc<[usize]>,
}

impl Display for VarBitPackedData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "offset: {}, chunks: {}", self.offset, self.starts.len() - 1)
    }
}

impl ArrayHash for VarBitPackedData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.offset.hash(state);
        self.packed.array_hash(state, accuracy);
        self.widths.array_hash(state, accuracy);
    }
}

impl ArrayEq for VarBitPackedData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.offset == other.offset
            && self.packed.array_eq(&other.packed, accuracy)
            && self.widths.array_eq(&other.widths, accuracy)
    }
}

/// Words of a `bytes`-byte type that one chunk packed at `width` bits occupies.
fn chunk_words(width: u8, bytes: usize) -> usize {
    128 * usize::from(width) / bytes
}

impl VarBitPackedData {
    fn try_new(
        packed: BufferHandle,
        widths: BufferHandle,
        offset: u16,
        ptype: PType,
        len: usize,
    ) -> VortexResult<Self> {
        vortex_ensure!(ptype.is_int(), "VarBitPacked requires an integer type, got {ptype}");
        vortex_ensure!(
            usize::from(offset) < FL_CHUNK_SIZE,
            "VarBitPacked offset must be less than {FL_CHUNK_SIZE}, got {offset}"
        );
        let bytes = ptype.byte_width();
        let widths_host = widths.as_host();
        let num_chunks = (usize::from(offset) + len).div_ceil(FL_CHUNK_SIZE);
        vortex_ensure!(
            widths_host.len() == num_chunks,
            "VarBitPacked expects {num_chunks} chunk widths, got {}",
            widths_host.len()
        );
        let mut starts = Vec::with_capacity(num_chunks + 1);
        let mut total = 0usize;
        for &width in widths_host.as_slice() {
            vortex_ensure!(
                usize::from(width) <= bytes * 8,
                "VarBitPacked width {width} exceeds {ptype}"
            );
            starts.push(total);
            total += chunk_words(width, bytes);
        }
        starts.push(total);
        vortex_ensure!(
            packed.len() == total * bytes,
            "VarBitPacked expects {} packed bytes, got {}",
            total * bytes,
            packed.len()
        );
        Ok(Self {
            offset,
            packed,
            widths,
            starts: starts.into(),
        })
    }

    fn widths(&self) -> &[u8] {
        self.widths.as_host().as_slice()
    }

    fn packed_words<P: NativePType>(&self) -> &[P] {
        let bytes = self.packed.as_host();
        // SAFETY: the buffer holds whole words of `P`, aligned on construction.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len() / size_of::<P>()) }
    }

    /// Call `f` with each chunk's packed words, width, and the range of output positions it
    /// covers, counted from the start of the first chunk.
    pub(crate) fn for_each_chunk<P: NativePType>(
        &self,
        len: usize,
        mut f: impl FnMut(&[P], usize, Range<usize>),
    ) {
        let words = self.packed_words::<P>();
        let end = usize::from(self.offset) + len;
        for (c, &width) in self.widths().iter().enumerate() {
            let start = c * FL_CHUNK_SIZE;
            f(
                &words[self.starts[c]..self.starts[c + 1]],
                usize::from(width),
                start..(start + FL_CHUNK_SIZE).min(end),
            );
        }
    }
}

pub trait VarBitPackedArrayExt: VarBitPackedArraySlotsExt {
    /// The position of the first element within the first chunk.
    fn offset(&self) -> u16 {
        self.offset
    }

    /// The bit width of each chunk.
    fn chunk_widths(&self) -> &[u8] {
        self.widths()
    }

    fn validity(&self) -> Validity {
        child_to_validity(self.validity_child(), self.as_ref().dtype().nullability())
    }
}

impl<T: TypedArrayRef<VarBitPacked>> VarBitPackedArrayExt for T {}

impl VarBitPacked {
    /// Construct from packed chunks and their widths.
    pub fn try_new(
        packed: BufferHandle,
        widths: BufferHandle,
        ptype: PType,
        validity: Validity,
        len: usize,
        offset: u16,
    ) -> VortexResult<VarBitPackedArray> {
        let dtype = DType::Primitive(ptype, validity.nullability());
        let mut slots = ArraySlots::with_capacity(1);
        slots.push(validity_to_child(&validity, len));
        let data = VarBitPackedData::try_new(packed, widths, offset, ptype, len)?;
        Array::try_from_parts(ArrayParts::new(VarBitPacked, dtype, len, data).with_slots(slots))
    }

    /// Pack every chunk of `array` at the width of its largest value.
    ///
    /// Values are packed as their unsigned bit patterns, so negative values pack at the full
    /// width and the encoding is lossless for any integers. Null positions pack as zero.
    pub fn encode(array: &PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<VarBitPackedArray> {
        let validity = array.validity()?;
        let mask = validity.execute_mask(array.len(), ctx)?;
        match_each_integer_ptype!(array.ptype(), |T| {
            let values: &[<T as PhysicalPType>::Physical] = {
                let values = array.as_slice::<T>();
                // SAFETY: `T::Physical` is `T` with the same size and alignment.
                unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len()) }
            };
            let (packed, widths) = pack(values, &mask);
            Self::try_new(
                BufferHandle::new_host(packed.into_byte_buffer()),
                BufferHandle::new_host(widths.into_byte_buffer()),
                array.ptype(),
                validity,
                array.len(),
                0,
            )
        })
    }
}

/// Pack each chunk of `values` at its own width, with nulls as zero.
fn pack<P>(values: &[P], mask: &vortex_mask::Mask) -> (Buffer<P>, Buffer<u8>)
where
    P: NativePType + PrimInt + BitPacking,
{
    let num_chunks = values.len().div_ceil(FL_CHUNK_SIZE);
    let mut widths = BufferMut::<u8>::with_capacity(num_chunks);
    let mut packed = BufferMut::<P>::empty();
    let mut chunk = [P::zero(); FL_CHUNK_SIZE];
    let bits = size_of::<P>() * 8;
    for (c, values) in values.chunks(FL_CHUNK_SIZE).enumerate() {
        let start = c * FL_CHUNK_SIZE;
        chunk.fill(P::zero());
        let mut max = P::zero();
        for (j, &v) in values.iter().enumerate() {
            if mask.value(start + j) {
                chunk[j] = v;
                max = max.max(v);
            }
        }
        let width = bits - max.leading_zeros() as usize;
        let words = chunk_words(width as u8, size_of::<P>());
        let at = packed.len();
        packed.extend(std::iter::repeat_n(P::zero(), words));
        if width > 0 {
            // SAFETY: `chunk` is a whole chunk and the destination has room for it at `width`.
            unsafe { BitPacking::unchecked_pack(width, &chunk, &mut packed[at..]) };
        }
        widths.push(width as u8);
    }
    (packed.freeze(), widths.freeze())
}

/// Unpack one chunk of `width` bits into `dst`, which holds a whole chunk.
#[inline]
pub(crate) fn unpack_chunk<P: BitPacking + Default>(words: &[P], width: usize, dst: &mut [P]) {
    if width == 0 {
        dst.fill(P::default());
    } else {
        // SAFETY: `words` holds one chunk at `width` and `dst` is a whole chunk.
        unsafe { BitPacking::unchecked_unpack(width, words, dst) };
    }
}

fn decompress(array: ArrayView<'_, VarBitPacked>, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    match_each_integer_ptype!(array.dtype().as_ptype(), |T| {
        decompress_typed::<T>(array, ctx)
    })
}

fn decompress_typed<T: PhysicalPType<Physical: BitPacking + PrimInt>>(
    array: ArrayView<'_, VarBitPacked>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let len = array.len();
    let offset = usize::from(array.offset);
    let mut values = BufferMut::<T>::with_capacity_in(len, ctx.allocator().clone());
    let output: &mut [MaybeUninit<T::Physical>] = {
        let spare = &mut values.spare_capacity_mut()[..len];
        // SAFETY: `T::Physical` is `T` with the same size and alignment.
        unsafe { std::mem::transmute(spare) }
    };
    let mut scratch = [<T::Physical as num_traits::Zero>::zero(); FL_CHUNK_SIZE];
    array.data().for_each_chunk::<T::Physical>(len, |words, width, range| {
        let skip = offset.saturating_sub(range.start);
        let dst = &mut output[range.start + skip - offset..range.end - offset];
        if dst.len() == FL_CHUNK_SIZE {
            // SAFETY: every value of `dst` is written by the unpack.
            let dst = unsafe { std::mem::transmute::<&mut [MaybeUninit<T::Physical>], &mut [T::Physical]>(dst) };
            unpack_chunk(words, width, dst);
        } else {
            unpack_chunk(words, width, &mut scratch);
            for (o, &v) in dst.iter_mut().zip(&scratch[skip..range.len()]) {
                o.write(v);
            }
        }
    });
    // SAFETY: the loop above initialized every value.
    unsafe { values.set_len(len) };
    Ok(PrimitiveArray::new(
        values.freeze(),
        VarBitPackedArrayExt::validity(&array),
    ))
}

impl VTable for VarBitPacked {
    type TypedArrayData = VarBitPackedData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        var_bitpacked_id()
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        _slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(dtype.is_int(), "VarBitPacked requires an integer dtype, got {dtype}");
        VarBitPackedData::try_new(
            data.packed.clone(),
            data.widths.clone(),
            data.offset,
            dtype.as_ptype(),
            len,
        )
        .map(|_| ())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        2
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => array.data().packed.clone(),
            1 => array.data().widths.clone(),
            _ => vortex_panic!("VarBitPackedArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some("packed".to_string()),
            1 => Some("widths".to_string()),
            _ => None,
        }
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(buffers.len() == 2, "Expected 2 buffers, got {}", buffers.len());
        let data = VarBitPackedData::try_new(
            buffers[0].clone(),
            buffers[1].clone(),
            array.data().offset,
            array.dtype().as_ptype(),
            array.len(),
        )?;
        Ok(
            ArrayParts::new(self.clone(), array.dtype().clone(), array.len(), data)
                .with_slots(array.slots().iter().cloned().collect()),
        )
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        VarBitPackedSlots::NAMES[idx].to_string()
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        vortex_bail!("VarBitPacked serialization requires VarBitPackedPlugin")
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_bail!("VarBitPacked deserialization requires VarBitPackedPlugin")
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        require_validity!(array, VarBitPackedSlots::VALIDITY_CHILD);
        Ok(ExecutionResult::done(decompress(array.as_view(), ctx)?.into_array()))
    }
}

const RULES: ParentRuleSet<VarBitPacked> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&SliceReduceAdaptor(VarBitPacked))]);

impl SliceReduce for VarBitPacked {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let data = array.data();
        let start = usize::from(data.offset) + range.start;
        let end = usize::from(data.offset) + range.end;
        let chunks = start / FL_CHUNK_SIZE..end.div_ceil(FL_CHUNK_SIZE).max(start / FL_CHUNK_SIZE);
        let bytes = array.dtype().as_ptype().byte_width();
        let packed = data
            .packed
            .slice(data.starts[chunks.start] * bytes..data.starts[chunks.end] * bytes);
        let widths = data.widths.slice(chunks);
        let validity = VarBitPackedArrayExt::validity(&array).slice(range.clone())?;
        Ok(Some(
            VarBitPacked::try_new(
                packed,
                widths,
                array.dtype().as_ptype(),
                validity,
                range.len(),
                u16::try_from(start % FL_CHUNK_SIZE)?,
            )?
            .into_array(),
        ))
    }
}

impl ValidityVTable<VarBitPacked> for VarBitPacked {
    fn validity(array: ArrayView<'_, VarBitPacked>) -> VortexResult<Validity> {
        Ok(VarBitPackedArrayExt::validity(&array))
    }
}

impl OperationsVTable<VarBitPacked> for VarBitPacked {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, VarBitPacked>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let position = usize::from(array.offset) + index;
        let chunk = position / FL_CHUNK_SIZE;
        let data = array.data();
        let width = usize::from(data.widths()[chunk]);
        Ok(match_each_integer_ptype!(array.dtype().as_ptype(), |T| {
            type P = <T as PhysicalPType>::Physical;
            let words = &data.packed_words::<P>()[data.starts[chunk]..data.starts[chunk + 1]];
            let value: P = if width == 0 {
                0
            } else {
                // SAFETY: `words` holds one chunk at `width`, and the index is within it.
                unsafe { BitPacking::unchecked_unpack_single(width, words, position % FL_CHUNK_SIZE) }
            };
            let value: T = value.as_();
            Scalar::primitive::<T>(value, array.dtype().nullability())
        }))
    }
}
