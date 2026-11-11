// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Byte encodings shared by the index kinds' columns.

use roaring::RoaringBitmap;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

/// Serialize a roaring bitmap in its portable format.
pub(crate) fn bitmap_to_bytes(bitmap: &RoaringBitmap) -> VortexResult<Vec<u8>> {
    let mut bytes = Vec::with_capacity(bitmap.serialized_size());
    bitmap
        .serialize_into(&mut bytes)
        .map_err(|err| vortex_err!("Failed to serialize roaring bitmap: {err}"))?;
    Ok(bytes)
}

/// Deserialize a roaring bitmap written by [`bitmap_to_bytes`].
pub(crate) fn bitmap_from_bytes(bytes: &[u8]) -> VortexResult<RoaringBitmap> {
    RoaringBitmap::deserialize_from(bytes)
        .map_err(|err| vortex_err!("Failed to deserialize roaring bitmap: {err}"))
}

/// Pack `u32`s as LEB128 varints: one byte for each value below 128, which term frequencies
/// almost always are.
#[expect(
    clippy::cast_possible_truncation,
    reason = "each byte keeps the low seven bits on purpose"
)]
pub(crate) fn u32s_to_varints(values: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len());
    for &v in values {
        let mut v = v;
        while v >= 0x80 {
            out.push((v as u8) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }
    out
}

/// Unpack varints written by [`u32s_to_varints`].
pub(crate) fn u32s_from_varints(bytes: &[u8]) -> VortexResult<Vec<u32>> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut value = 0u32;
    let mut shift = 0u32;
    for &b in bytes {
        vortex_ensure!(shift < 32, "varint overflows u32");
        value |= u32::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            out.push(value);
            value = 0;
            shift = 0;
        } else {
            shift += 7;
        }
    }
    vortex_ensure_eq!(shift, 0, "truncated varint");
    Ok(out)
}

/// Pack `f32`s little-endian.
pub(crate) fn f32s_to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Unpack little-endian `f32`s written by [`f32s_to_bytes`].
pub(crate) fn f32s_from_bytes(bytes: &[u8]) -> VortexResult<Vec<f32>> {
    vortex_ensure_eq!(bytes.len() % 4, 0, "packed f32 buffer has a ragged tail");
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

/// A canonical struct chunk of an index child, with typed accessors for its fields.
pub(crate) struct IndexRowsChunk {
    rows: StructArray,
}

impl IndexRowsChunk {
    pub(crate) fn new(chunk: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        Ok(Self {
            rows: chunk.execute::<StructArray>(ctx)?,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.rows.len()
    }

    pub(crate) fn field(&self, name: &str) -> VortexResult<ArrayRef> {
        Ok(self.rows.unmasked_field_by_name(name)?.clone())
    }

    /// Every value of a binary field, in row order.
    pub(crate) fn binary_field(
        &self,
        name: &str,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Vec<ByteBuffer>> {
        let values = self.field(name)?.execute::<VarBinViewArray>(ctx)?;
        Ok((0..values.len()).map(|idx| values.bytes_at(idx)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varints_roundtrip() -> VortexResult<()> {
        let values = vec![0, 1, 127, 128, 300, 16_384, u32::MAX];
        assert_eq!(u32s_from_varints(&u32s_to_varints(&values))?, values);
        assert_eq!(u32s_to_varints(&[1, 2, 3]).len(), 3);
        Ok(())
    }
}
