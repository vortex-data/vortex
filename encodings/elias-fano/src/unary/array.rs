// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMeta;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

/// The parts of a [`Unary`](crate::Unary) array: its bits.
#[derive(Clone, Debug)]
pub struct UnaryData {
    pub(super) bits: BufferHandle,
    pub(super) meta: BitBufferMeta,
}

impl Display for UnaryData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "bits: {}", self.meta.len())
    }
}

/// The number of bits `len` values take with `num_ones` set bits, if a `usize` can count them.
pub(super) fn bit_len(len: usize, num_ones: u64) -> Option<usize> {
    u64::try_from(len)
        .ok()?
        .checked_add(num_ones)
        .and_then(|bits| usize::try_from(bits).ok())
}

impl UnaryData {
    /// The parts holding `bits`.
    pub(super) fn from_bit_buffer(bits: BitBuffer) -> Self {
        // `shrink_offset` drops the whole bytes before and after the bits, so the buffer holds only
        // the bytes they use and the offset is below 8, as `BitBufferMeta` requires.
        let (offset, len, buffer) = bits.shrink_offset().into_inner();
        Self {
            bits: BufferHandle::new_host(buffer),
            meta: BitBufferMeta::new(offset, len),
        }
    }

    /// Rebuild the parts from a stored buffer whose bits start `offset` bits in. Only the offset
    /// and the buffer's size are checked here. The readers rebuild the bits with
    /// [`BitBuffer::new_with_offset`], whose own check can pass a short buffer on 32-bit, so this
    /// one must be exact. [`Array::try_from_parts`](vortex_array::Array::try_from_parts) validates
    /// the rest.
    pub(super) fn from_buffer(
        buffer: BufferHandle,
        num_ones: u64,
        offset: usize,
        len: usize,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            offset < 8,
            "UnaryArray bit offset must be below 8, got {offset}"
        );
        let bits = bit_len(len, num_ones).ok_or_else(|| {
            vortex_err!(
                "UnaryArray of {len} values with {num_ones} set bits overflows a usize bit count"
            )
        })?;
        // Saturating can only undercount what the buffer holds, so it never passes one too short.
        vortex_ensure!(
            buffer
                .len()
                .saturating_mul(8)
                .checked_sub(offset)
                .is_some_and(|room| room >= bits),
            "UnaryArray needs {bits} bits from bit {offset}, but its buffer holds {} bytes",
            buffer.len()
        );
        Ok(Self {
            bits: buffer,
            meta: BitBufferMeta::new(offset, bits),
        })
    }

    /// Check the parts against an array of `len` values of `dtype`.
    ///
    /// The dtype and the bit length are always checked. When the bits are on the host, this also
    /// checks the last bit and counts the set bits, and if the values could sum past the ptype's
    /// maximum, walks them once to check that each value fits. The readers rely on this and check
    /// no value themselves.
    pub(super) fn validate(&self, dtype: &DType, len: usize) -> VortexResult<()> {
        let DType::Primitive(ptype, NonNullable) = dtype else {
            vortex_bail!("UnaryArray requires a non-nullable unsigned integer dtype, got {dtype}");
        };
        vortex_ensure!(
            ptype.is_unsigned_int(),
            "UnaryArray requires a non-nullable unsigned integer dtype, got {dtype}"
        );
        let num_ones = self.meta.len().checked_sub(len).ok_or_else(|| {
            vortex_err!(
                "UnaryArray of {len} values has only {} bits",
                self.meta.len()
            )
        })?;
        if !self.bits.is_on_host() {
            return Ok(());
        }
        let bits = self.to_bit_buffer();
        // An unset last bit closes the last value, and leaves an empty array no bits at all.
        vortex_ensure!(
            bits.is_empty() || !bits.value(bits.len() - 1),
            "UnaryArray bits must end with an unset bit"
        );
        // The linear checks run last. `num_ones` set bits among `len + num_ones` leave exactly
        // `len` unset, one closing each value.
        vortex_ensure_eq!(bits.true_count(), num_ones);
        let max = ptype.max_value_as_u64();
        if u64::try_from(num_ones)? > max {
            let mut fits = true;
            self.for_each_value(|value| {
                fits &= u64::try_from(value).is_ok_and(|value| value <= max);
            });
            vortex_ensure!(fits, "UnaryArray holds a value that does not fit {ptype}");
        }
        Ok(())
    }

    /// The bits. Panics if the buffer is on a device.
    pub(super) fn to_bit_buffer(&self) -> BitBuffer {
        BitBuffer::new_with_offset(
            self.bits.as_host().clone(),
            self.meta.len(),
            self.meta.offset(),
        )
    }

    /// Call `f` with every value, in order, from one walk over the unset bits.
    pub(super) fn for_each_value(&self, mut f: impl FnMut(usize)) {
        let mut start = 0;
        self.to_bit_buffer().for_each_unset_index(|end| {
            f(end - start);
            start = end + 1;
        });
    }

    /// The value at `index`, found by scanning from the first bit to the unset bit before it, then
    /// on to its own closing bit.
    pub(super) fn value_at(&self, index: usize) -> VortexResult<u64> {
        let missing = || vortex_err!("UnaryArray has no unset bit for index {index}");
        let bits = self.to_bit_buffer();
        let start = match index.checked_sub(1) {
            Some(previous) => bits.select_unset(previous).ok_or_else(missing)? + 1,
            None => 0,
        };
        // The value is the run of set bits from `start` to the next unset bit, so this continues
        // from `start` rather than scanning from the first bit again.
        let value = bits.slice(start..).select_unset(0).ok_or_else(missing)?;
        Ok(u64::try_from(value)?)
    }
}
