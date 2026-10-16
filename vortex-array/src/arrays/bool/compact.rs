// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMeta;
use vortex_error::VortexResult;

use crate::arrays::BoolArray;
use crate::buffer::BufferHandle;

impl BoolArray {
    /// Trims the packed bit buffer to the bytes backing the array's visible bits.
    ///
    /// Arrays built over a shared buffer, such as one decoded from a file segment, can hold
    /// trailing bytes that no read can see but that `nbytes` and serialization still pay for. The
    /// trim is zero-copy and keeps the leading bit offset, so only whole bytes are dropped.
    pub fn trim_bits(&self) -> VortexResult<Self> {
        let byte_len = BitBufferMeta::new(self.meta.offset(), self.len()).byte_len();

        let bits = match self.bits.as_host_opt() {
            Some(host) => BufferHandle::new_host(host.slice_unaligned(..byte_len)),
            None => self.bits.slice(0..byte_len),
        };

        Self::try_new_from_handle(bits, self.meta.offset(), self.len(), self.validity()?)
    }

    /// Trims the packed bit buffer as [`trim_bits`](Self::trim_bits) does, and also drops the
    /// leading bit offset when it costs a byte.
    ///
    /// The bits of `len` values fit in `len.div_ceil(8)` bytes, but a nonzero bit offset can push
    /// the last of them into one more byte. That case copies the bits to start at offset zero, so
    /// the buffer holds exactly the bytes its values need. Bits on a device are only trimmed.
    pub fn trim_bits_exact(&self) -> VortexResult<Self> {
        let trimmed = self.trim_bits()?;
        let packed_len = trimmed.len().div_ceil(8);
        if BitBufferMeta::new(trimmed.meta.offset(), trimmed.len()).byte_len() == packed_len {
            return Ok(trimmed);
        }
        let Some(host) = trimmed.bits.as_host_opt() else {
            return Ok(trimmed);
        };
        let bits = BitBuffer::new_with_offset(host.clone(), trimmed.len(), trimmed.meta.offset());
        let aligned = BitBuffer::collect_bool(bits.len(), |index| bits.value(index));
        Ok(Self::new(aligned, trimmed.validity()?))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;

    use crate::Canonical;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::assert_arrays_eq;
    use crate::buffer::BufferHandle;
    use crate::validity::Validity;

    #[rstest]
    #[case(0, 16, Validity::NonNullable)]
    #[case(0, 13, Validity::NonNullable)]
    #[case(3, 10, Validity::AllValid)]
    #[case(7, 1, Validity::NonNullable)]
    #[case(0, 0, Validity::NonNullable)]
    fn trims_to_visible_bytes(
        #[case] offset: usize,
        #[case] len: usize,
        #[case] validity: Validity,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let bytes: ByteBuffer = (0..64u8).collect();
        let handle = BufferHandle::new_host(bytes.clone());

        let trimmed =
            BoolArray::try_new_from_handle(handle, offset, len, validity.clone())?.trim_bits()?;

        assert_eq!(trimmed.bits.len(), (offset + len).div_ceil(8));
        let expected = BoolArray::new(BitBuffer::new_with_offset(bytes, len, offset), validity);
        assert_arrays_eq!(trimmed, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn canonical_compact_trims_bits() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let handle = BufferHandle::new_host((0..64u8).collect());
        let array = BoolArray::try_new_from_handle(handle, 0, 20, Validity::NonNullable)?;

        let trimmed = Canonical::Bool(array).compact(&mut ctx)?.into_bool();

        assert_eq!(trimmed.bits.len(), 3);
        Ok(())
    }
}
