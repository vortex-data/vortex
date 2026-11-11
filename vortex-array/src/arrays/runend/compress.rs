// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::arrays::runend::iter::slice_relative_ends;
use crate::buffer::BufferHandle;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::match_each_decimal_value_type;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// Run-end encode a `PrimitiveArray`, returning a tuple of `(ends, values)`.
pub fn runend_encode(
    array: ArrayView<Primitive>,
    ctx: &mut ExecutionCtx,
) -> (PrimitiveArray, ArrayRef) {
    let validity = match array
        .validity()
        .vortex_expect("run-end validity should be derivable")
    {
        Validity::NonNullable => None,
        Validity::AllValid => None,
        Validity::AllInvalid => {
            // We can trivially return an all-null REE array
            let ends = PrimitiveArray::new(buffer![array.len() as u64], Validity::NonNullable);
            ends.statistics()
                .set(Stat::IsStrictSorted, Precision::Exact(true.into()));
            return (
                ends,
                ConstantArray::new(Scalar::null(array.dtype().clone()), 1).into_array(),
            );
        }
        Validity::Array(a) => {
            let bool_array = a
                .execute::<BoolArray>(ctx)
                .vortex_expect("validity array must be convertible to bool");
            Some(bool_array.to_bit_buffer())
        }
    };

    let (ends, values) = match validity {
        None => {
            match_each_native_ptype!(array.ptype(), |P| {
                let (ends, values) = runend_encode_primitive(array.as_slice::<P>());
                (
                    PrimitiveArray::new(ends, Validity::NonNullable),
                    PrimitiveArray::new(values, array.dtype().nullability().into()).into_array(),
                )
            })
        }
        Some(validity) => {
            match_each_native_ptype!(array.ptype(), |P| {
                let (ends, values) =
                    runend_encode_nullable_primitive(array.as_slice::<P>(), validity);
                (
                    PrimitiveArray::new(ends, Validity::NonNullable),
                    values.into_array(),
                )
            })
        }
    };

    let ends = ends
        .narrow(ctx)
        .vortex_expect("Ends must succeed downcasting");

    ends.statistics()
        .set(Stat::IsStrictSorted, Precision::Exact(true.into()));

    (ends, values)
}

fn runend_encode_primitive<T: NativePType>(elements: &[T]) -> (Buffer<u64>, Buffer<T>) {
    let mut ends = BufferMut::empty();
    let mut values = BufferMut::empty();

    if elements.is_empty() {
        return (ends.freeze(), values.freeze());
    }

    // Run-end encode the values
    let mut prev = elements[0];
    let mut end = 1;
    for &e in elements.iter().skip(1) {
        if e != prev {
            ends.push(end);
            values.push(prev);
        }
        prev = e;
        end += 1;
    }
    ends.push(end);
    values.push(prev);

    (ends.freeze(), values.freeze())
}

fn runend_encode_nullable_primitive<T: NativePType>(
    elements: &[T],
    element_validity: BitBuffer,
) -> (Buffer<u64>, PrimitiveArray) {
    let mut ends = BufferMut::empty();
    let mut values = BufferMut::empty();
    let mut validity = BitBufferMut::with_capacity(values.capacity());

    if elements.is_empty() {
        return (
            ends.freeze(),
            PrimitiveArray::new(
                values,
                Validity::Array(BoolArray::from(validity.freeze()).into_array()),
            ),
        );
    }

    // Run-end encode the values
    let mut prev = element_validity.value(0).then(|| elements[0]);
    let mut end = 1;
    for e in elements
        .iter()
        .zip(element_validity.iter())
        .map(|(&e, is_valid)| is_valid.then_some(e))
        .skip(1)
    {
        if e != prev {
            ends.push(end);
            match prev {
                None => {
                    validity.append(false);
                    values.push(T::default());
                }
                Some(p) => {
                    validity.append(true);
                    values.push(p);
                }
            }
        }
        prev = e;
        end += 1;
    }
    ends.push(end);

    match prev {
        None => {
            validity.append(false);
            values.push(T::default());
        }
        Some(p) => {
            validity.append(true);
            values.push(p);
        }
    }

    (
        ends.freeze(),
        PrimitiveArray::new(values, Validity::from(validity.freeze())),
    )
}

/// Run-end encode a `BoolArray` without nulls, returning a tuple of `(ends, values)`.
///
/// The values child keeps the input's nullability.
///
/// # Errors
///
/// Returns an error if the array contains nulls.
pub fn runend_encode_bool(
    array: ArrayView<Bool>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(PrimitiveArray, ArrayRef)> {
    let has_nulls = match array.validity()? {
        Validity::NonNullable | Validity::AllValid => false,
        Validity::AllInvalid => !array.is_empty(),
        Validity::Array(validity) => {
            validity
                .execute::<BoolArray>(ctx)?
                .to_bit_buffer()
                .true_count()
                != array.len()
        }
    };
    if has_nulls {
        vortex_bail!("run-end encoding of booleans with nulls is not supported");
    }

    let bits = array.to_bit_buffer();

    let mut ends = BufferMut::<u64>::empty();
    for_each_transition_word(&bits, |base, mut transitions| {
        while transitions != 0 {
            ends.push((base + transitions.trailing_zeros() as usize) as u64);
            transitions &= transitions - 1;
        }
    });
    if !bits.is_empty() {
        ends.push(bits.len() as u64);
    }

    // Runs alternate between `true` and `false`, so the first bit gives all of the values.
    let first = !bits.is_empty() && bits.value(0);
    let values = BitBuffer::collect_bool(ends.len(), |run| first ^ (run % 2 == 1));

    let ends = PrimitiveArray::new(ends.freeze(), Validity::NonNullable).narrow(ctx)?;
    ends.statistics()
        .set(Stat::IsStrictSorted, Precision::Exact(true.into()));
    let values = BoolArray::new(values, array.dtype().nullability().into()).into_array();
    Ok((ends, values))
}

/// Count the runs of equal values in a bitmap, a word at a time.
pub fn bool_run_count(bits: &BitBuffer) -> usize {
    if bits.is_empty() {
        return 0;
    }

    let mut transitions = 0;
    for_each_transition_word(bits, |_, word| transitions += word.count_ones() as usize);
    transitions + 1
}

/// Calls `f(base, transitions)` for each 64-bit word of `bits`, where bit `j` of `transitions` is
/// set when bit `base + j` differs from bit `base + j - 1`.
///
/// Bit 0 of the bitmap never counts as a transition.
fn for_each_transition_word(bits: &BitBuffer, mut f: impl FnMut(usize, u64)) {
    if bits.is_empty() {
        return;
    }

    let chunks = bits.chunks();

    // Seed the carry with the first bit, so that bit 0 compares equal to its predecessor.
    let mut carry = u64::from(bits.value(0));
    let mut base = 0;

    for word in chunks.iter() {
        f(base, word ^ ((word << 1) | carry));
        carry = word >> 63;
        base += 64;
    }

    let remainder_len = chunks.remainder_len();
    if remainder_len > 0 {
        let word = chunks.remainder_bits();
        let mask = u64::MAX >> (64 - remainder_len);
        f(base, (word ^ ((word << 1) | carry)) & mask);
    }
}

pub fn runend_decode_primitive(
    ends: PrimitiveArray,
    values: PrimitiveArray,
    offset: usize,
    length: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let validity_mask = values
        .as_ref()
        .validity()?
        .execute_mask(values.as_ref().len(), ctx)?;
    if matches!(validity_mask, Mask::AllFalse(_)) {
        return Ok(
            ConstantArray::new(Scalar::null(values.dtype().as_nullable()), length).into_array(),
        );
    }
    Ok(match_each_native_ptype!(values.ptype(), |P| {
        match_each_unsigned_integer_ptype!(ends.ptype(), |E| {
            runend_decode_typed_primitive(
                slice_relative_ends(ends.as_slice::<E>(), offset, length),
                values.as_slice::<P>(),
                validity_mask,
                values.dtype().nullability(),
                length,
                ctx.allocator(),
            )
            .into_array()
        })
    }))
}

/// Decode a run-end encoded slice of values into a flat `Buffer<T>` and `Validity`.
///
/// This is the core decode loop shared by primitive and varbinview run-end decoding.
fn runend_decode_slice<T: Copy + Default>(
    run_ends: impl Iterator<Item = usize>,
    values: &[T],
    values_validity: Mask,
    values_nullability: Nullability,
    length: usize,
    allocator: &BufferAllocatorRef,
) -> (Buffer<T>, Validity) {
    match values_validity {
        Mask::AllTrue(_) => {
            let mut decoded = RunWriter::new(length, allocator);
            for (end, &value) in run_ends.zip_eq(values) {
                decoded.write_run(value, end);
            }
            (decoded.finish(), values_nullability.into())
        }
        Mask::AllFalse(_) => (Buffer::<T>::zeroed(length), Validity::AllInvalid),
        Mask::Values(mask) => {
            // Nulls are usually rare and clustered, so start from all-valid and clear each
            // maximal streak of null runs once, rather than appending validity per run.
            let mut decoded = RunWriter::new(length, allocator);
            let mut decoded_validity = BitBufferMut::full(true, length);
            let mut null_start = None;
            for (end, (&value, is_valid)) in
                run_ends.zip_eq(values.iter().zip(mask.bit_buffer().iter()))
            {
                let start = decoded.len();
                decoded.write_run(if is_valid { value } else { T::default() }, end);
                match (is_valid, null_start) {
                    (false, None) => null_start = Some(start),
                    (true, Some(null_from)) => {
                        // SAFETY: null_from <= start <= length == decoded_validity.len()
                        unsafe { decoded_validity.fill_range_unchecked(null_from, start, false) };
                        null_start = None;
                    }
                    _ => {}
                }
            }
            let decoded = decoded.finish();
            if let Some(null_from) = null_start {
                // SAFETY: null_from <= decoded.len() <= length == decoded_validity.len()
                unsafe { decoded_validity.fill_range_unchecked(null_from, decoded.len(), false) };
            }
            decoded_validity.truncate(decoded.len());
            (decoded, Validity::from(decoded_validity.freeze()))
        }
    }
}

/// Minimum bytes in a [`RunWriter`] block.
const BLOCK_BYTES: usize = 32;

/// Writes runs of values into a buffer of up to `length` elements.
///
/// Runs are usually short, and filling each one with a loop sized to its length costs more than
/// the fill itself. Instead a run that fits in a fixed block writes the whole block, which the
/// compiler lowers to a few vector stores, and the next run overwrites whatever went past its
/// end. Longer runs are filled exactly. The buffer has a block of spare capacity so that the last
/// run's block stays in bounds.
struct RunWriter<T> {
    buffer: BufferMut<T>,
    len: usize,
    length: usize,
}

impl<T: Copy> RunWriter<T> {
    /// Elements in a block: enough to cover most short runs, at least 8 elements and 32 bytes.
    const BLOCK: usize = if size_of::<T>() == 0 || size_of::<T>() >= BLOCK_BYTES / 8 {
        8
    } else {
        BLOCK_BYTES / size_of::<T>()
    };

    fn new(length: usize, allocator: &BufferAllocatorRef) -> Self {
        Self {
            buffer: BufferMut::with_capacity_in(length + Self::BLOCK, allocator.clone()),
            len: 0,
            length,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    /// Fill `self.len()..end` with `value`.
    #[inline]
    fn write_run(&mut self, value: T, end: usize) {
        assert!(
            end >= self.len,
            "Runend ends must be monotonic, got {end} after {}",
            self.len
        );
        assert!(
            end <= self.length,
            "Runend end must be less than overall length"
        );
        // SAFETY: `self.len <= end <= length`, and the buffer holds `length + BLOCK` elements, so
        // both the block and the remainder of the run are in bounds.
        unsafe {
            let dst = self.buffer.as_mut_ptr().add(self.len);
            let run_len = end - self.len;
            if run_len <= Self::BLOCK {
                for i in 0..Self::BLOCK {
                    dst.add(i).write(value);
                }
            } else {
                for i in 0..run_len {
                    dst.add(i).write(value);
                }
            }
        }
        self.len = end;
    }

    fn finish(mut self) -> Buffer<T> {
        // SAFETY: every element below `self.len` was written by a run.
        unsafe { self.buffer.set_len(self.len) };
        self.buffer.freeze()
    }
}

pub fn runend_decode_typed_primitive<T: NativePType>(
    run_ends: impl Iterator<Item = usize>,
    values: &[T],
    values_validity: Mask,
    values_nullability: Nullability,
    length: usize,
    allocator: &BufferAllocatorRef,
) -> PrimitiveArray {
    let (decoded, validity) = runend_decode_slice(
        run_ends,
        values,
        values_validity,
        values_nullability,
        length,
        allocator,
    );
    PrimitiveArray::new(decoded, validity)
}

/// Decode a run-end encoded decimal array by expanding its native values.
pub fn runend_decode_decimal(
    ends: PrimitiveArray,
    values: DecimalArray,
    offset: usize,
    length: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let validity_mask = values
        .as_ref()
        .validity()?
        .execute_mask(values.as_ref().len(), ctx)?;
    if matches!(validity_mask, Mask::AllFalse(_)) {
        return Ok(
            ConstantArray::new(Scalar::null(values.dtype().as_nullable()), length).into_array(),
        );
    }

    Ok(match_each_decimal_value_type!(values.values_type(), |D| {
        let (decoded, validity) = match_each_unsigned_integer_ptype!(ends.ptype(), |E| {
            runend_decode_slice(
                slice_relative_ends(ends.as_slice::<E>(), offset, length),
                values.buffer::<D>().as_slice(),
                validity_mask,
                values.dtype().nullability(),
                length,
                ctx.allocator(),
            )
        });
        DecimalArray::new(decoded, values.decimal_dtype(), validity).into_array()
    }))
}

/// Decode a run-end encoded VarBinView array by expanding views directly.
pub fn runend_decode_varbinview(
    ends: PrimitiveArray,
    values: VarBinViewArray,
    offset: usize,
    length: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let validity_mask = values
        .as_ref()
        .validity()?
        .execute_mask(values.as_ref().len(), ctx)?;
    if matches!(validity_mask, Mask::AllFalse(_)) {
        return Ok(
            ConstantArray::new(Scalar::null(values.dtype().as_nullable()), length).into_array(),
        );
    }
    let views = values.views();

    let (decoded_views, validity) = match_each_unsigned_integer_ptype!(ends.ptype(), |E| {
        runend_decode_slice(
            slice_relative_ends(ends.as_slice::<E>(), offset, length),
            views,
            validity_mask,
            values.dtype().nullability(),
            length,
            ctx.allocator(),
        )
    });

    let parts = values.into_data_parts();
    let view_handle = BufferHandle::new_host(decoded_views.into_byte_buffer());

    // SAFETY: we are expanding views from a valid VarBinViewArray with the same
    // buffers, so all buffer indices and offsets remain valid.
    Ok(unsafe {
        VarBinViewArray::new_handle_unchecked(view_handle, parts.buffers, parts.dtype, validity)
    }
    .into_array())
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::RunEnd;
    use crate::arrays::bool::BoolArrayExt;
    use crate::arrays::runend::compress::bool_run_count;
    use crate::arrays::runend::compress::runend_decode_primitive;
    use crate::arrays::runend::compress::runend_encode;
    use crate::assert_arrays_eq;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    /// Bits alternate in runs of `run`, with lengths on 64-bit word boundaries. `valid` adds a
    /// validity bitmap that is all valid (`true`) or has one null (`false`).
    #[rstest]
    #[case(0, 1, None)]
    #[case(1, 1, None)]
    #[case(63, 62, None)]
    #[case(64, 63, None)]
    #[case(65, 64, None)]
    #[case(130, 1, None)]
    #[case(300, 61, Some(true))]
    #[case(300, 61, Some(false))]
    fn encode_bool(
        #[case] len: usize,
        #[case] run: usize,
        #[case] valid: Option<bool>,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let bits: Vec<bool> = (0..len).map(|i| (i / run) % 2 == 1).collect();
        // Array-backed even when all valid, which `Validity::from(BitBuffer)` would collapse.
        let validity = valid.map_or(Validity::NonNullable, |v| {
            Validity::Array(BoolArray::from_iter((0..len).map(|i| v || i != 1)).into_array())
        });
        let arr = BoolArray::new(BitBuffer::from(bits.clone()), validity);
        let runs = |b: &[bool]| b.len().min(1) + b.windows(2).filter(|w| w[0] != w[1]).count();
        for offset in [0, 1].into_iter().filter(|&o| o <= len) {
            let sliced = arr.to_bit_buffer().slice(offset..);
            assert_eq!(bool_run_count(&sliced), runs(&bits[offset..]));

            let sliced = BoolArray::new(sliced, Validity::NonNullable);
            let encoded = RunEnd::encode(sliced.clone().into_array(), &mut ctx)?;
            assert_arrays_eq!(encoded, sliced, &mut ctx);
        }

        let encoded = RunEnd::encode(arr.clone().into_array(), &mut ctx);
        if valid == Some(false) {
            assert!(encoded.is_err());
        } else {
            assert_arrays_eq!(encoded?, arr, &mut ctx);
        }
        Ok(())
    }

    #[test]
    fn encode() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let arr = PrimitiveArray::from_iter([1i32, 1, 2, 2, 2, 3, 3, 3, 3, 3]);
        let (ends, values) = runend_encode(arr.as_view(), &mut ctx);

        let expected_ends = PrimitiveArray::from_iter(vec![2u8, 5, 10]);
        assert_arrays_eq!(ends, expected_ends, &mut ctx);
        let expected_values = PrimitiveArray::from_iter(vec![1i32, 2, 3]);
        assert_arrays_eq!(values, expected_values, &mut ctx);
        Ok(())
    }

    #[test]
    fn encode_nullable() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let arr = PrimitiveArray::new(
            buffer![1i32, 1, 2, 2, 2, 3, 3, 3, 3, 3],
            Validity::from(BitBuffer::from(vec![
                true, true, false, false, true, true, true, true, false, false,
            ])),
        );
        let (ends, values) = runend_encode(arr.as_view(), &mut ctx);

        let expected_ends = PrimitiveArray::from_iter(vec![2u8, 4, 5, 8, 10]);
        assert_arrays_eq!(ends, expected_ends, &mut ctx);
        let expected_values =
            PrimitiveArray::from_option_iter(vec![Some(1i32), None, Some(2), Some(3), None]);
        assert_arrays_eq!(values, expected_values, &mut ctx);
        Ok(())
    }

    #[test]
    fn encode_all_null() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let arr = PrimitiveArray::new(
            buffer![0, 0, 0, 0, 0],
            Validity::from(BitBuffer::new_unset(5)),
        );
        let (ends, values) = runend_encode(arr.as_view(), &mut ctx);

        let expected_ends = PrimitiveArray::from_iter(vec![5u64]);
        assert_arrays_eq!(ends, expected_ends, &mut ctx);
        let expected_values = PrimitiveArray::from_option_iter(vec![Option::<i32>::None]);
        assert_arrays_eq!(values, expected_values, &mut ctx);
        Ok(())
    }

    #[test]
    fn decode() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let ends = PrimitiveArray::from_iter([2u32, 5, 10]);
        let values = PrimitiveArray::from_iter([1i32, 2, 3]);
        let decoded = runend_decode_primitive(ends, values, 0, 10, &mut ctx)?;

        let expected = PrimitiveArray::from_iter(vec![1i32, 1, 2, 2, 2, 3, 3, 3, 3, 3]);
        assert_arrays_eq!(decoded, expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::middle_nulls(&[Some(1), None, None, Some(2)], 0, 10)]
    #[case::leading_and_trailing_nulls(&[None, Some(1), Some(2), None], 0, 10)]
    #[case::all_null(&[None, None, None, None], 0, 10)]
    #[case::offset_and_trailing_null(&[None, Some(1), Some(2), None], 1, 8)]
    fn decode_nullable(
        #[case] values: &[Option<i32>],
        #[case] offset: usize,
        #[case] length: usize,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let ends = [2u32, 5, 7, 10];
        let expected = PrimitiveArray::from_option_iter(
            (offset..offset + length)
                .map(|row| values[ends.iter().position(|&end| row < end as usize).unwrap_or(0)]),
        );
        let decoded = runend_decode_primitive(
            PrimitiveArray::from_iter(ends),
            PrimitiveArray::from_option_iter(values.iter().copied()),
            offset,
            length,
            &mut ctx,
        )?;
        assert_arrays_eq!(decoded, expected, &mut ctx);
        Ok(())
    }
}
