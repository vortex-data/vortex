// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;
use std::sync::Arc;

use itertools::Itertools as _;
use num_traits::AsPrimitive;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Columnar;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::PiecewiseSequence;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBin;
use crate::arrays::VarBinArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::dict::TakeExecute;
use crate::arrays::piecewise_sequence::constant_unsigned_usize;
use crate::arrays::piecewise_sequence::maybe_contiguous_slices;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::arrays::varbin::VarBinArrayExt;
use crate::arrays::varbin::VarBinArraySlotsExt;
use crate::arrays::varbinview::BinaryView;
use crate::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::PType;
use crate::dtype::UnsignedPType;
use crate::executor::ExecutionCtx;
use crate::match_each_integer_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// The widened offset type used for a taken `VarBinArray`: offsets are widened to at least 32 bits
/// (to avoid overflow) while preserving signedness, so a signed result stays Arrow-compatible.
fn taken_offset_ptype(offsets_ptype: PType) -> PType {
    match offsets_ptype {
        PType::U8 | PType::U16 | PType::U32 => PType::U32,
        PType::U64 => PType::U64,
        PType::I8 | PType::I16 | PType::I32 => PType::I32,
        PType::I64 => PType::I64,
        _ => unreachable!("invalid PType for offsets"),
    }
}

// "take" can have offsets which don't fit in 32 bits. "signed" is needed for
// Arrow interoperability.
enum Offsets {
    U32 {
        offsets: BufferMut<u32>,
        signed: bool,
    },
    U64 {
        offsets: BufferMut<u64>,
        signed: bool,
    },
}

impl Offsets {
    fn with_capacity(offset_ptype: PType, capacity: usize) -> Self {
        let signed = offset_ptype.is_signed_int();
        match offset_ptype {
            PType::U64 | PType::I64 => Self::U64 {
                offsets: BufferMut::with_capacity(capacity),
                signed,
            },
            _ => Self::U32 {
                offsets: BufferMut::with_capacity(capacity),
                signed,
            },
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::U32 { offsets, .. } => offsets.len(),
            Self::U64 { offsets, .. } => offsets.len(),
        }
    }

    fn push(&mut self, offset: usize) {
        let (offsets, signed) = match self {
            Self::U64 { offsets, .. } => {
                offsets.push(offset as u64);
                return;
            }
            Self::U32 { offsets, signed } => (offsets, *signed),
        };
        let max_narrow = if signed {
            i32::MAX as usize
        } else {
            u32::MAX as usize
        };
        if offset <= max_narrow {
            offsets.push(u32::try_from(offset).vortex_expect("offset fits u32"));
            return;
        }

        // Copying the buffer may look scary but in fact it isn't.
        // If we don't copy-and-widen, we need to know target width
        // beforehand. This is approach taken in ListArray, where first pass
        // iterates over offsets and determines target width, and second offset
        // copies the offset to the buffer. Both have similar performance
        // Note this happens only if 32-bit offset overflows which means array
        // must hold >4GB of data which is quite rare.
        let mut wide = BufferMut::<u64>::with_capacity(offsets.capacity() + 1);
        wide.extend(offsets.iter().map(|&o| u64::from(o)));
        wide.push(offset as u64);
        *self = Self::U64 {
            offsets: wide,
            signed,
        };
    }

    fn into_array(self) -> ArrayRef {
        let (offsets, out_offset_ptype) = match self {
            Self::U32 { offsets, signed } => (
                PrimitiveArray::new(offsets.freeze(), Validity::NonNullable),
                if signed { PType::I32 } else { PType::U32 },
            ),
            Self::U64 { offsets, signed } => (
                PrimitiveArray::new(offsets.freeze(), Validity::NonNullable),
                if signed { PType::I64 } else { PType::U64 },
            ),
        };
        offsets.reinterpret_cast(out_offset_ptype).into_array()
    }
}

impl TakeExecute for VarBin {
    fn take(
        array: ArrayView<'_, VarBin>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let offsets = array.offsets().clone().execute::<PrimitiveArray>(ctx)?;
        let offsets = offsets.reinterpret_cast(offsets.ptype().to_unsigned());
        let last_offset = match_each_unsigned_integer_ptype!(offsets.ptype(), |O| {
            offsets.as_slice::<O>().last().map_or(0usize, |&o| o.as_())
        });

        // VarBinView can't hold this buffer, so we can't canonicalize and
        // take() (take panics). Convert to VarBin
        if last_offset > MAX_BUFFER_LEN {
            return Ok(Some(take_varbin(array, indices, ctx)?.into_array()));
        }

        let data = array.bytes().clone();
        let dtype = array
            .dtype()
            .clone()
            .union_nullability(indices.dtype().nullability());
        let validity = array.validity()?.take(indices)?;

        let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
        let indices_mask = indices
            .as_ref()
            .validity()?
            .execute_mask(indices.as_ref().len(), ctx)?;

        let views = match_each_unsigned_integer_ptype!(offsets.ptype(), |O| {
            match_each_integer_ptype!(indices.ptype(), |I| {
                take_views(
                    offsets.as_slice::<O>(),
                    data.as_slice(),
                    indices.as_slice::<I>(),
                    &indices_mask,
                )
            })
        });

        // SAFETY: every view references buffer 0 which is inside shared data buffer
        unsafe {
            Ok(Some(
                VarBinViewArray::new_unchecked(views, Arc::from([data]), dtype, validity)
                    .into_array(),
            ))
        }
    }
}

fn take_views<O: UnsignedPType, I: IntegerPType + AsPrimitive<usize>>(
    offsets: &[O],
    data: &[u8],
    indices: &[I],
    mask: &Mask,
) -> Buffer<BinaryView> {
    let build = |idx: usize| -> BinaryView {
        // SAFETY: caller verified idx and idx + 1 are in bounds
        let start: usize = unsafe { *offsets.get_unchecked(idx) }.as_();
        let stop: usize = unsafe { *offsets.get_unchecked(idx + 1) }.as_();
        let value = &data[start..stop];
        let len = stop - start;

        // Caller guarantees every offset is <= MAX_BUFFER_LEN
        let start: u32 = start.as_();
        if len > BinaryView::MAX_INLINED_SIZE {
            let mut prefix = [0u8; 4];
            prefix.copy_from_slice(&value[..4]);
            let len: u32 = len.as_();
            BinaryView::new_ref(len, prefix, 0, start)
        } else {
            BinaryView::make_view(value, 0, start)
        }
    };

    match mask.bit_buffer() {
        AllOr::All => {
            let iter = indices.iter().map(|i| build(i.as_()));
            let Some(&first) = indices.first() else {
                return Buffer::from_trusted_len_iter(iter);
            };
            let (min, max) = indices
                .iter()
                .fold((first, first), |(lo, hi), &x| (lo.min(x), hi.max(x)));
            assert!(
                min >= I::zero() && max.as_() < offsets.len() - 1,
                "take index out of bounds"
            );
            Buffer::from_trusted_len_iter(iter)
        }
        AllOr::None => {
            Buffer::from_trusted_len_iter(iter::repeat_n(BinaryView::default(), indices.len()))
        }
        AllOr::Some(buffer) => {
            let Some(max_element) = (offsets.len() - 1).checked_sub(1) else {
                assert_eq!(mask.true_count(), 0, "take index out of bounds");
                return Buffer::from_trusted_len_iter(iter::repeat_n(
                    BinaryView::default(),
                    indices.len(),
                ));
            };
            let mut violations = 0usize;
            let views = Buffer::from_trusted_len_iter(buffer.iter().zip(indices.iter()).map(
                |(valid, i)| {
                    let keep = usize::from(valid).wrapping_neg();
                    let masked = i.as_() & keep;
                    let clamped = masked.min(max_element);
                    violations += usize::from(masked != clamped);
                    let view = build(clamped);
                    if valid { view } else { BinaryView::default() }
                },
            ));
            assert_eq!(violations, 0, "take index out of bounds");
            views
        }
    }
}

/// Take from a VarBin. Referenced bytes are copied
pub fn take_varbin(
    array: ArrayView<'_, VarBin>,
    indices: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<VarBinArray> {
    if let Some(piecewise_indices) = indices.as_opt::<PiecewiseSequence>()
        && let Some(taken) = take_contiguous_ranges(array, piecewise_indices, indices, ctx)?
    {
        return Ok(taken);
    }

    let offsets = array.offsets().clone().execute::<PrimitiveArray>(ctx)?;
    let data = array.bytes();
    let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
    let dtype = array
        .dtype()
        .clone()
        .union_nullability(indices.dtype().nullability());
    let array_validity = array
        .varbin_validity()
        .execute_mask(array.as_ref().len(), ctx)?;
    let indices_validity = indices
        .as_ref()
        .validity()?
        .execute_mask(indices.as_ref().len(), ctx)?;

    // Offsets and indices are non-negative; read them through their unsigned reinterpretations
    // so we only monomorphize over the 4 unsigned widths each (4x4 instead of 8x8). On take,
    // offsets get widened to either 32- or 64-bit (to avoid overflow); the built output offsets
    // are reinterpreted back to `out_offset_ptype` to preserve the result's offset signedness.
    let out_offset_ptype = taken_offset_ptype(offsets.ptype());
    let offsets = offsets.reinterpret_cast(offsets.ptype().to_unsigned());
    let indices = indices.reinterpret_cast(indices.ptype().to_unsigned());

    match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
        match_each_unsigned_integer_ptype!(offsets.ptype(), |O| {
            take::<I, O>(
                dtype,
                offsets.as_slice::<O>(),
                data.as_slice(),
                indices.as_slice::<I>(),
                array_validity,
                indices_validity,
                out_offset_ptype,
            )
        })
    })
}

fn take_contiguous_ranges(
    array: ArrayView<'_, VarBin>,
    indices: ArrayView<'_, PiecewiseSequence>,
    indices_ref: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<VarBinArray>> {
    let Some((starts, lengths)) = maybe_contiguous_slices(indices, ctx)? else {
        return Ok(None);
    };
    let offsets = array.offsets().clone().execute::<PrimitiveArray>(ctx)?;
    let out_offset_ptype = taken_offset_ptype(offsets.ptype());
    let offsets = offsets.reinterpret_cast(offsets.ptype().to_unsigned());
    let bytes = array.bytes();
    let data = bytes.as_slice();
    let dtype = array.dtype().clone();
    let output_len = indices_ref.len();

    let result = match lengths {
        Columnar::Constant(lengths) => {
            let length = constant_unsigned_usize(&lengths);
            gather_slices_constant_dispatch(
                &starts,
                length,
                &offsets,
                data,
                output_len,
                out_offset_ptype,
            )?
        }
        Columnar::Canonical(lengths) => {
            let lengths = lengths.into_primitive();
            gather_slices_dispatch(
                &starts,
                &lengths,
                &offsets,
                data,
                output_len,
                out_offset_ptype,
            )?
        }
    };

    let validity = array.validity()?.take(indices_ref)?;

    // SAFETY: output offsets are built from valid input offsets, start at zero, are monotonically
    // non-decreasing, and the copied data buffer has exactly the referenced byte length.
    unsafe {
        Ok(Some(VarBinArray::new_unchecked(
            result.offsets,
            result.data.freeze(),
            dtype,
            validity,
        )))
    }
}

fn take<Index: IntegerPType, Offset: IntegerPType>(
    dtype: DType,
    offsets: &[Offset],
    data: &[u8],
    indices: &[Index],
    validity_mask: Mask,
    indices_validity_mask: Mask,
    out_offset_ptype: PType,
) -> VortexResult<VarBinArray> {
    if !validity_mask.all_true() || !indices_validity_mask.all_true() {
        return Ok(take_nullable::<Index, Offset>(
            dtype,
            offsets,
            data,
            indices,
            validity_mask,
            indices_validity_mask,
            out_offset_ptype,
        ));
    }

    let mut new_offsets = Offsets::with_capacity(out_offset_ptype, indices.len() + 1);
    new_offsets.push(0);
    let mut current_offset = 0usize;

    if let Some(&first) = indices.first() {
        let (min, max) = indices
            .iter()
            .fold((first, first), |(lo, hi), &x| (lo.min(x), hi.max(x)));
        assert!(
            min >= Index::zero() && max.as_() < offsets.len() - 1,
            "take index out of bounds"
        );
    }

    for &idx in indices {
        let idx: usize = idx.as_();
        // SAFETY: idx is in bounds, checked before
        let start = unsafe { *offsets.get_unchecked(idx) };
        let stop = unsafe { *offsets.get_unchecked(idx + 1) };

        current_offset += (stop - start)
            .to_usize()
            .vortex_expect("Failed to cast offset to usize");
        new_offsets.push(current_offset);
    }

    let mut new_data = ByteBufferMut::with_capacity(current_offset);

    for &idx in indices {
        let idx: usize = idx.as_();
        // SAFETY: idx is in bounds, checked before
        let start = unsafe { *offsets.get_unchecked(idx) }
            .to_usize()
            .vortex_expect("Failed to cast max offset to usize");
        let stop = unsafe { *offsets.get_unchecked(idx + 1) }
            .to_usize()
            .vortex_expect("Failed to cast max offset to usize");
        new_data.extend_from_slice(&data[start..stop]);
    }

    let array_validity = Validity::from(dtype.nullability());
    let new_offsets = new_offsets.into_array();

    // Safety:
    // All variants of VarBinArray are satisfied here.
    unsafe {
        Ok(VarBinArray::new_unchecked(
            new_offsets,
            new_data.freeze(),
            dtype,
            array_validity,
        ))
    }
}

struct GatheredPiecewiseVarBin {
    offsets: ArrayRef,
    data: ByteBufferMut,
}

fn gather_slices_constant_dispatch(
    starts: &PrimitiveArray,
    length: usize,
    offsets: &PrimitiveArray,
    data: &[u8],
    output_len: usize,
    out_offset_ptype: PType,
) -> VortexResult<GatheredPiecewiseVarBin> {
    match_each_unsigned_integer_ptype!(starts.ptype(), |S| {
        gather_slices_constant_start_dispatch::<S>(
            starts,
            length,
            offsets,
            data,
            output_len,
            out_offset_ptype,
        )
    })
}

fn gather_slices_constant_start_dispatch<S>(
    starts: &PrimitiveArray,
    length: usize,
    offsets: &PrimitiveArray,
    data: &[u8],
    output_len: usize,
    out_offset_ptype: PType,
) -> VortexResult<GatheredPiecewiseVarBin>
where
    S: UnsignedPType,
{
    match offsets.ptype() {
        PType::U8 => gather_slices_constant_length::<S, u8>(
            offsets.as_slice::<u8>(),
            data,
            starts.as_slice::<S>(),
            length,
            output_len,
            out_offset_ptype,
        ),
        PType::U16 => gather_slices_constant_length::<S, u16>(
            offsets.as_slice::<u16>(),
            data,
            starts.as_slice::<S>(),
            length,
            output_len,
            out_offset_ptype,
        ),
        PType::U32 => gather_slices_constant_length::<S, u32>(
            offsets.as_slice::<u32>(),
            data,
            starts.as_slice::<S>(),
            length,
            output_len,
            out_offset_ptype,
        ),
        PType::U64 => gather_slices_constant_length::<S, u64>(
            offsets.as_slice::<u64>(),
            data,
            starts.as_slice::<S>(),
            length,
            output_len,
            out_offset_ptype,
        ),
        _ => unreachable!("offsets were reinterpreted to an unsigned integer ptype"),
    }
}

fn gather_slices_dispatch(
    starts: &PrimitiveArray,
    lengths: &PrimitiveArray,
    offsets: &PrimitiveArray,
    data: &[u8],
    output_len: usize,
    out_offset_ptype: PType,
) -> VortexResult<GatheredPiecewiseVarBin> {
    match_each_unsigned_integer_ptype!(starts.ptype(), |S| {
        gather_slices_start_dispatch::<S>(
            starts,
            lengths,
            offsets,
            data,
            output_len,
            out_offset_ptype,
        )
    })
}

fn gather_slices_start_dispatch<S>(
    starts: &PrimitiveArray,
    lengths: &PrimitiveArray,
    offsets: &PrimitiveArray,
    data: &[u8],
    output_len: usize,
    out_offset_ptype: PType,
) -> VortexResult<GatheredPiecewiseVarBin>
where
    S: UnsignedPType,
{
    match_each_unsigned_integer_ptype!(lengths.ptype(), |L| {
        gather_slices_start_length_dispatch::<S, L>(
            starts,
            lengths,
            offsets,
            data,
            output_len,
            out_offset_ptype,
        )
    })
}

fn gather_slices_start_length_dispatch<S, L>(
    starts: &PrimitiveArray,
    lengths: &PrimitiveArray,
    offsets: &PrimitiveArray,
    data: &[u8],
    output_len: usize,
    out_offset_ptype: PType,
) -> VortexResult<GatheredPiecewiseVarBin>
where
    S: UnsignedPType,
    L: UnsignedPType,
{
    match offsets.ptype() {
        PType::U8 => gather_slices::<S, L, u8>(
            offsets.as_slice::<u8>(),
            data,
            starts.as_slice::<S>(),
            lengths.as_slice::<L>(),
            output_len,
            out_offset_ptype,
        ),
        PType::U16 => gather_slices::<S, L, u16>(
            offsets.as_slice::<u16>(),
            data,
            starts.as_slice::<S>(),
            lengths.as_slice::<L>(),
            output_len,
            out_offset_ptype,
        ),
        PType::U32 => gather_slices::<S, L, u32>(
            offsets.as_slice::<u32>(),
            data,
            starts.as_slice::<S>(),
            lengths.as_slice::<L>(),
            output_len,
            out_offset_ptype,
        ),
        PType::U64 => gather_slices::<S, L, u64>(
            offsets.as_slice::<u64>(),
            data,
            starts.as_slice::<S>(),
            lengths.as_slice::<L>(),
            output_len,
            out_offset_ptype,
        ),
        _ => unreachable!("offsets were reinterpreted to an unsigned integer ptype"),
    }
}

fn gather_slices_constant_length<S, Offset>(
    offsets: &[Offset],
    data: &[u8],
    starts: &[S],
    length: usize,
    output_len: usize,
    out_offset_ptype: PType,
) -> VortexResult<GatheredPiecewiseVarBin>
where
    S: UnsignedPType,
    Offset: IntegerPType,
{
    let computed_len = starts
        .len()
        .checked_mul(length)
        .ok_or_else(|| vortex_err!("PiecewiseSequenceArray output length overflows usize"))?;
    vortex_ensure_eq!(
        computed_len,
        output_len,
        "PiecewiseSequenceArray expanded length does not match declared length",
    );

    let mut new_offsets = Offsets::with_capacity(out_offset_ptype, output_len + 1);
    new_offsets.push(0);
    let mut output_bytes = 0usize;

    for start in starts {
        let start = start.as_();
        if length == 0 {
            continue;
        }

        let offset_range = &offsets[start..][..=length];
        let byte_start = offset_range[0].as_();
        let byte_end = offset_range[length].as_();
        vortex_ensure!(
            byte_start <= byte_end && byte_end <= data.len(),
            "VarBin offsets range {byte_start}..{byte_end} exceeds data length {}",
            data.len()
        );

        for &offset in &offset_range[1..] {
            let offset = offset.as_();
            let relative = offset.checked_sub(byte_start).ok_or_else(|| {
                vortex_err!("VarBin offsets are not monotonic at offset {offset}")
            })?;
            let output_offset = output_bytes.checked_add(relative).ok_or_else(|| {
                vortex_err!("PiecewiseSequence VarBin output byte length overflow")
            })?;
            new_offsets.push(output_offset);
        }

        output_bytes = output_bytes
            .checked_add(byte_end - byte_start)
            .ok_or_else(|| vortex_err!("PiecewiseSequence VarBin output byte length overflow"))?;
    }

    let mut new_data = ByteBufferMut::with_capacity(output_bytes);
    let spare = &mut new_data.spare_capacity_mut()[..output_bytes];
    let mut cursor = 0usize;
    for start in starts {
        let start = start.as_();
        if length == 0 {
            continue;
        }

        let offset_range = &offsets[start..][..=length];
        let byte_start = offset_range[0].as_();
        let byte_end = offset_range[length].as_();
        let src = &data[byte_start..byte_end];
        spare[cursor..][..src.len()].write_copy_of_slice(src);
        cursor += src.len();
    }
    // SAFETY: the loop initialized the prefix `0..cursor` of the spare capacity.
    unsafe { new_data.set_len(cursor) };
    vortex_ensure_eq!(
        new_data.len(),
        output_bytes,
        "PiecewiseSequenceArray gathered byte length does not match declared byte length",
    );

    let offsets = new_offsets.into_array();
    Ok(GatheredPiecewiseVarBin {
        offsets,
        data: new_data,
    })
}

fn gather_slices<S, L, Offset>(
    offsets: &[Offset],
    data: &[u8],
    starts: &[S],
    lengths: &[L],
    output_len: usize,
    out_offset_ptype: PType,
) -> VortexResult<GatheredPiecewiseVarBin>
where
    S: UnsignedPType,
    L: UnsignedPType,
    Offset: IntegerPType,
{
    let mut new_offsets = Offsets::with_capacity(out_offset_ptype, output_len + 1);
    new_offsets.push(0);
    let mut output_bytes = 0usize;

    for (&start, &length) in starts.iter().zip_eq(lengths) {
        let start = start.as_();
        let length = length.as_();
        if length == 0 {
            continue;
        }

        let offset_range = &offsets[start..][..=length];
        let byte_start = offset_range[0].as_();
        let byte_end = offset_range[length].as_();
        vortex_ensure!(
            byte_start <= byte_end && byte_end <= data.len(),
            "VarBin offsets range {byte_start}..{byte_end} exceeds data length {}",
            data.len()
        );

        for &offset in &offset_range[1..] {
            let offset = offset.as_();
            let relative = offset.checked_sub(byte_start).ok_or_else(|| {
                vortex_err!("VarBin offsets are not monotonic at offset {offset}")
            })?;
            let output_offset = output_bytes.checked_add(relative).ok_or_else(|| {
                vortex_err!("PiecewiseSequence VarBin output byte length overflow")
            })?;
            new_offsets.push(output_offset);
        }

        output_bytes = output_bytes
            .checked_add(byte_end - byte_start)
            .ok_or_else(|| vortex_err!("PiecewiseSequence VarBin output byte length overflow"))?;
    }
    vortex_ensure_eq!(
        new_offsets.len(),
        output_len + 1,
        "PiecewiseSequenceArray expanded offsets length does not match declared length plus one",
    );

    let mut new_data = ByteBufferMut::with_capacity(output_bytes);
    let spare = &mut new_data.spare_capacity_mut()[..output_bytes];
    let mut cursor = 0usize;
    for (&start, &length) in starts.iter().zip_eq(lengths) {
        let start = start.as_();
        let length = length.as_();
        if length == 0 {
            continue;
        }

        let offset_range = &offsets[start..][..=length];
        let byte_start = offset_range[0].as_();
        let byte_end = offset_range[length].as_();
        let src = &data[byte_start..byte_end];
        spare[cursor..][..src.len()].write_copy_of_slice(src);
        cursor += src.len();
    }
    // SAFETY: the loop initialized the prefix `0..cursor` of the spare capacity.
    unsafe { new_data.set_len(cursor) };
    vortex_ensure_eq!(
        new_data.len(),
        output_bytes,
        "PiecewiseSequenceArray gathered byte length does not match declared byte length",
    );

    let offsets = new_offsets.into_array();
    Ok(GatheredPiecewiseVarBin {
        offsets,
        data: new_data,
    })
}

fn take_nullable<Index: IntegerPType, Offset: IntegerPType>(
    dtype: DType,
    offsets: &[Offset],
    data: &[u8],
    indices: &[Index],
    data_validity: Mask,
    indices_validity: Mask,
    out_offset_ptype: PType,
) -> VarBinArray {
    let mut new_offsets = Offsets::with_capacity(out_offset_ptype, indices.len() + 1);
    new_offsets.push(0);
    let mut current_offset = 0usize;

    let mut validity_buffer = BitBufferMut::with_capacity(indices.len());

    // Convert indices once and store valid ones with their positions
    let mut valid_indices = Vec::with_capacity(indices.len());

    let max_element = offsets.len().saturating_sub(2);
    assert!(
        offsets.len() >= 2 || indices_validity.true_count() == 0,
        "take index out of bounds"
    );
    let mut violations = 0usize;

    // First pass: calculate offsets and validity
    for (data_idx, index_valid) in indices.iter().zip(indices_validity.iter()) {
        if !index_valid {
            validity_buffer.append(false);
            new_offsets.push(current_offset);
            continue;
        }
        let raw: usize = data_idx.as_();
        let data_idx_usize = raw.min(max_element);
        violations += usize::from(raw != data_idx_usize);
        if data_validity.value(data_idx_usize) {
            validity_buffer.append(true);
            let start = unsafe { *offsets.get_unchecked(data_idx_usize) };
            let stop = unsafe { *offsets.get_unchecked(data_idx_usize + 1) };
            current_offset += (stop - start)
                .to_usize()
                .vortex_expect("Failed to cast offset to usize");
            new_offsets.push(current_offset);
            valid_indices.push(data_idx_usize);
        } else {
            validity_buffer.append(false);
            new_offsets.push(current_offset);
        }
    }
    assert_eq!(violations, 0, "take index out of bounds");

    let mut new_data = ByteBufferMut::with_capacity(current_offset);

    // Second pass: copy data for valid indices only
    for data_idx in valid_indices {
        let start = unsafe { *offsets.get_unchecked(data_idx) }
            .to_usize()
            .vortex_expect("Failed to cast max offset to usize");
        let stop = unsafe { *offsets.get_unchecked(data_idx + 1) }
            .to_usize()
            .vortex_expect("Failed to cast max offset to usize");
        new_data.extend_from_slice(&data[start..stop]);
    }

    let array_validity = Validity::from(validity_buffer.freeze());
    let new_offsets = new_offsets.into_array();

    // Safety:
    // All variants of VarBinArray are satisfied here.
    unsafe { VarBinArray::new_unchecked(new_offsets, new_data.freeze(), dtype, array_validity) }
}

#[cfg(test)]
mod tests {
    use std::iter;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::varbin::compute::take::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn test_null_take() {
        let arr = VarBinArray::from_iter([Some("h")], DType::Utf8(Nullability::NonNullable));

        let idx1: PrimitiveArray = (0..1).collect();

        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(
            arr.take(idx1.into_array()).unwrap(),
            VarBinViewArray::from_iter_str(["h"]),
            &mut ctx
        );

        let idx2: PrimitiveArray = PrimitiveArray::from_option_iter(vec![Some(0)]);

        assert_arrays_eq!(
            arr.take(idx2.into_array()).unwrap(),
            VarBinViewArray::from_iter_nullable_str([Some("h")]),
            &mut ctx
        );
    }

    #[rstest]
    #[case(VarBinArray::from_iter(
        ["hello", "world", "test", "data", "array"].map(Some),
        DType::Utf8(Nullability::NonNullable),
    ))]
    #[case(VarBinArray::from_iter(
        [Some("hello"), None, Some("test"), Some("data"), None],
        DType::Utf8(Nullability::Nullable),
    ))]
    #[case(VarBinArray::from_iter(
        [b"hello".as_slice(), b"world", b"test", b"data", b"array"].map(Some),
        DType::Binary(Nullability::NonNullable),
    ))]
    #[case(VarBinArray::from_iter(["single"].map(Some), DType::Utf8(Nullability::NonNullable)))]
    fn test_take_varbin_conformance(#[case] array: VarBinArray) {
        test_take_conformance(&array.into_array(), &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_take_overflow() {
        let mut ctx = SESSION.create_execution_ctx();
        let scream = iter::once("a").cycle().take(128).collect::<String>();
        let bytes = ByteBuffer::copy_from(scream.as_bytes());
        let offsets = buffer![0u8, 128u8].into_array();

        let array = VarBinArray::new(
            offsets,
            bytes,
            DType::Utf8(Nullability::NonNullable),
            Validity::NonNullable,
            &mut ctx,
        );

        let indices = buffer![0u32; 3].into_array();
        let taken = array.take(indices).unwrap();

        let expected = VarBinViewArray::from_iter(
            [Some(scream.clone()), Some(scream.clone()), Some(scream)],
            DType::Utf8(Nullability::NonNullable),
        );
        assert_arrays_eq!(taken, expected, &mut ctx);
    }

    #[rstest]
    #[case(PType::U32, u32::MAX as usize, PType::U32)]
    #[case(PType::U32, u32::MAX as usize + 1, PType::U64)]
    #[case(PType::I32, i32::MAX as usize, PType::I32)]
    #[case(PType::I32, i32::MAX as usize + 1, PType::I64)]
    #[case(PType::U64, 8, PType::U64)]
    #[case(PType::I64, 8, PType::I64)]
    fn test_offsets_widen(
        #[case] out_offset_ptype: PType,
        #[case] max_offset: usize,
        #[case] expected: PType,
    ) {
        let mut sink = super::Offsets::with_capacity(out_offset_ptype, 2);
        sink.push(0);
        sink.push(max_offset);
        let array = sink.into_array();
        assert_eq!(
            array.dtype(),
            &DType::Primitive(expected, Nullability::NonNullable)
        );
        assert_eq!(array.len(), 2);
        let mut ctx = SESSION.create_execution_ctx();
        for (index, value) in [0, max_offset as u64].into_iter().enumerate() {
            assert_eq!(
                array
                    .execute_scalar(index, &mut ctx)
                    .unwrap()
                    .as_primitive()
                    .as_::<u64>(),
                Some(value)
            );
        }
    }

    #[test]
    fn test_offset_widen_values() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let big = u32::MAX as usize + 5;
        let mut sink = super::Offsets::with_capacity(PType::U32, 3);
        sink.push(0);
        sink.push(7);
        sink.push(big);

        let array = sink.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(array.as_slice::<u64>(), &[0, 7, big as u64]);
        Ok(())
    }

    #[test]
    fn test_take_signed() {
        let mut ctx = SESSION.create_execution_ctx();
        let array = VarBinArray::new(
            buffer![0i32, 5, 10].into_array(),
            ByteBuffer::copy_from(b"helloworld"),
            DType::Utf8(Nullability::NonNullable),
            Validity::NonNullable,
            &mut ctx,
        );

        let taken = array.take(buffer![1u32, 0].into_array()).unwrap();

        let expected = VarBinViewArray::from_iter(
            [Some("world"), Some("hello")],
            DType::Utf8(Nullability::NonNullable),
        );
        assert_arrays_eq!(taken, expected, &mut ctx);
    }
}
