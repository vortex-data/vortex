// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `CanonicalMessage` wire format used to move canonical arrays across the host/guest
//! boundary in both directions.
//!
//! The format is a single contiguous, self-describing blob with inline buffer bytes so that one
//! copy moves an entire array. The guest SDK implements a byte-compatible encoder and decoder.
//! See `docs/design/wasm-encodings.md`.
//!
//! ## Nullability
//!
//! A nullable primitive is a values buffer plus a validity bitmap (LSB-first, 1 = valid). When
//! `validity == Bitmap` the message carries two buffers: buffer 0 is the values, buffer 1 is the
//! bitmap (`ceil(len / 8)` bytes). The values buffer always has an entry at every position;
//! null-ness lives entirely in the bitmap.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::NullArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::BitBuffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::abi::BUFFER_ENTRY_HEADER_LEN;
use crate::abi::MESSAGE_HEADER_LEN;
use crate::abi::MessageKind;
use crate::abi::MessageValidity;

fn ptype_to_u8(ptype: PType) -> u8 {
    match ptype {
        PType::U8 => 0,
        PType::U16 => 1,
        PType::U32 => 2,
        PType::U64 => 3,
        PType::I8 => 4,
        PType::I16 => 5,
        PType::I32 => 6,
        PType::I64 => 7,
        PType::F16 => 8,
        PType::F32 => 9,
        PType::F64 => 10,
    }
}

fn ptype_from_u8(value: u8) -> VortexResult<PType> {
    Ok(match value {
        0 => PType::U8,
        1 => PType::U16,
        2 => PType::U32,
        3 => PType::U64,
        4 => PType::I8,
        5 => PType::I16,
        6 => PType::I32,
        7 => PType::I64,
        8 => PType::F16,
        9 => PType::F32,
        10 => PType::F64,
        other => vortex_bail!("invalid CanonicalMessage ptype discriminant: {other}"),
    })
}

/// Smallest power-of-two alignment exponent that covers a `width`-byte scalar.
fn alignment_exponent(width: usize) -> u8 {
    debug_assert!(width.is_power_of_two());
    width.trailing_zeros() as u8
}

/// Encode an already-canonical array into a [`CanonicalMessage`] byte blob.
///
/// `ctx` is used to materialize a validity bitmap for nullable arrays. The first implementation
/// supports `Null` and `Primitive` (including bitmap validity). Other kinds return an error rather
/// than silently producing an unreadable message.
pub fn encode_canonical(canonical: &Canonical, ctx: &mut ExecutionCtx) -> VortexResult<Vec<u8>> {
    let mut out = Vec::new();
    match canonical {
        Canonical::Null(array) => {
            write_header(
                &mut out,
                MessageKind::Null,
                0,
                MessageValidity::NonNullable,
                array.len(),
                0,
                0,
            );
        }
        Canonical::Primitive(array) => {
            let ptype = array.ptype();
            let values = array.buffer_handle().to_host_sync();
            let bitmap = match array.validity()? {
                Validity::NonNullable | Validity::AllValid | Validity::AllInvalid => None,
                Validity::Array(_) => {
                    // Materialize the validity as a contiguous (offset-0) bitmap.
                    let mask = array.validity()?.execute_mask(array.len(), ctx)?;
                    let bits = mask.to_bit_buffer();
                    let nbytes = array.len().div_ceil(8);
                    Some(bits.inner().as_slice()[..nbytes].to_vec())
                }
            };
            let validity = match array.validity()? {
                Validity::NonNullable => MessageValidity::NonNullable,
                Validity::AllValid => MessageValidity::AllValid,
                Validity::AllInvalid => MessageValidity::AllInvalid,
                Validity::Array(_) => MessageValidity::Bitmap,
            };
            let nbuffers = if bitmap.is_some() { 2 } else { 1 };
            write_header(
                &mut out,
                MessageKind::Primitive,
                ptype_to_u8(ptype),
                validity,
                array.len(),
                nbuffers,
                0,
            );
            write_buffer(
                &mut out,
                alignment_exponent(ptype.byte_width()),
                values.as_ref(),
            );
            if let Some(bitmap) = bitmap {
                write_buffer(&mut out, 0, &bitmap);
            }
        }
        other => vortex_bail!(
            "CanonicalMessage abi v1 cannot encode canonical kind {:?}",
            std::mem::discriminant(other)
        ),
    }
    Ok(out)
}

fn write_header(
    out: &mut Vec<u8>,
    kind: MessageKind,
    ptype: u8,
    validity: MessageValidity,
    length: usize,
    nbuffers: u32,
    nchildren: u32,
) {
    out.push(kind as u8);
    out.push(ptype);
    out.push(validity as u8);
    out.push(0); // pad
    out.extend_from_slice(&(length as u64).to_le_bytes());
    out.extend_from_slice(&nbuffers.to_le_bytes());
    out.extend_from_slice(&nchildren.to_le_bytes());
    debug_assert_eq!(out.len(), MESSAGE_HEADER_LEN);
}

fn write_buffer(out: &mut Vec<u8>, alignment_exp: u8, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    out.push(alignment_exp);
    out.extend_from_slice(&[0u8; 7]); // pad
    out.extend_from_slice(bytes);
}

/// A parsed view over the fixed header of a [`CanonicalMessage`].
struct Header {
    kind: MessageKind,
    ptype: u8,
    validity: MessageValidity,
    length: usize,
    nbuffers: u32,
    nchildren: u32,
}

fn read_u32(bytes: &[u8], offset: usize) -> VortexResult<u32> {
    let end = offset + 4;
    vortex_ensure!(end <= bytes.len(), "CanonicalMessage truncated reading u32");
    Ok(u32::from_le_bytes(
        bytes[offset..end].try_into().expect("4 bytes"),
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> VortexResult<u64> {
    let end = offset + 8;
    vortex_ensure!(end <= bytes.len(), "CanonicalMessage truncated reading u64");
    Ok(u64::from_le_bytes(
        bytes[offset..end].try_into().expect("8 bytes"),
    ))
}

fn read_header(bytes: &[u8]) -> VortexResult<Header> {
    vortex_ensure!(
        bytes.len() >= MESSAGE_HEADER_LEN,
        "CanonicalMessage shorter than header"
    );
    let kind = MessageKind::from_u8(bytes[0])
        .ok_or_else(|| vortex_error::vortex_err!("invalid CanonicalMessage kind {}", bytes[0]))?;
    let validity = MessageValidity::from_u8(bytes[2]).ok_or_else(|| {
        vortex_error::vortex_err!("invalid CanonicalMessage validity {}", bytes[2])
    })?;
    Ok(Header {
        kind,
        ptype: bytes[1],
        validity,
        length: usize::try_from(read_u64(bytes, 4)?)?,
        nbuffers: read_u32(bytes, 12)?,
        nchildren: read_u32(bytes, 16)?,
    })
}

/// Read each buffer's inline bytes in order.
fn read_buffers<'a>(bytes: &'a [u8], header: &Header) -> VortexResult<Vec<&'a [u8]>> {
    let mut offset = MESSAGE_HEADER_LEN;
    let mut out = Vec::with_capacity(header.nbuffers as usize);
    for _ in 0..header.nbuffers {
        let len = usize::try_from(read_u64(bytes, offset)?)?;
        let data_start = offset + BUFFER_ENTRY_HEADER_LEN;
        let data_end = data_start + len;
        vortex_ensure!(
            data_end <= bytes.len(),
            "CanonicalMessage truncated reading buffer data"
        );
        out.push(&bytes[data_start..data_end]);
        offset = data_end;
    }
    Ok(out)
}

/// Decode a [`CanonicalMessage`] byte blob into a Vortex array.
pub fn decode_message(bytes: &[u8]) -> VortexResult<ArrayRef> {
    let header = read_header(bytes)?;
    match header.kind {
        MessageKind::Null => Ok(NullArray::new(header.length).into_array()),
        MessageKind::Primitive => {
            vortex_ensure!(
                header.nchildren == 0,
                "primitive CanonicalMessage must have no children"
            );
            let expected_buffers = if header.validity == MessageValidity::Bitmap {
                2
            } else {
                1
            };
            vortex_ensure!(
                header.nbuffers == expected_buffers,
                "primitive CanonicalMessage with validity {:?} must have {expected_buffers} buffers, got {}",
                header.validity,
                header.nbuffers
            );
            let buffers = read_buffers(bytes, &header)?;
            let ptype = ptype_from_u8(header.ptype)?;
            let values =
                ByteBuffer::copy_from_aligned(buffers[0], Alignment::new(ptype.byte_width()));
            let validity = match header.validity {
                MessageValidity::NonNullable => Validity::NonNullable,
                MessageValidity::AllValid => Validity::AllValid,
                MessageValidity::AllInvalid => Validity::AllInvalid,
                MessageValidity::Bitmap => {
                    let bitmap = ByteBuffer::copy_from(buffers[1]);
                    Validity::from(BitBuffer::new(bitmap, header.length))
                }
            };
            Ok(PrimitiveArray::from_byte_buffer(values, ptype, validity).into_array())
        }
        other => vortex_bail!("CanonicalMessage abi v1 cannot decode kind {:?}", other),
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;

    #[test]
    fn primitive_round_trip() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = PrimitiveArray::new(buffer![1u32, 2, 3, 4, 5], Validity::NonNullable);
        let bytes = encode_canonical(&Canonical::Primitive(array), &mut ctx)?;
        let decoded = decode_message(&bytes)?;
        assert_eq!(decoded.len(), 5);
        let expected: Vec<u8> = [1u32, 2, 3, 4, 5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(decoded.buffers()[0].as_ref(), expected.as_slice());
        Ok(())
    }

    #[test]
    fn nullable_primitive_round_trip() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        // Positions 1 and 4 are null.
        let validity = Validity::from_iter([true, false, true, true, false]);
        let array = PrimitiveArray::new(buffer![10i64, 99, 30, 40, 99], validity);

        let bytes = encode_canonical(&Canonical::Primitive(array), &mut ctx)?;
        let decoded = decode_message(&bytes)?;

        assert_eq!(decoded.len(), 5);
        let bits = decoded
            .validity()?
            .execute_mask(5, &mut ctx)?
            .to_bit_buffer();
        let valid: Vec<bool> = (0..5).map(|i| bits.value(i)).collect();
        assert_eq!(valid, vec![true, false, true, true, false]);
        // Values at valid positions survive the round trip.
        let values: Vec<u8> = [10i64, 99, 30, 40, 99]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(decoded.buffers()[0].as_ref(), values.as_slice());
        Ok(())
    }
}
