// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Building and parsing the `CanonicalMessage` wire format from inside a guest kernel.
//!
//! Byte-compatible with `vortex-wasm`'s `message` module.

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::abi::BUFFER_ENTRY_HEADER_LEN;
use crate::abi::MESSAGE_HEADER_LEN;
use crate::abi::MessageKind;
use crate::abi::MessageValidity;
use crate::abi::PType;

/// Builder for a single [`CanonicalMessage`].
pub struct MessageBuilder {
    out: Vec<u8>,
}

impl MessageBuilder {
    /// Start a message with the given header fields. Append buffers with [`Self::buffer`].
    pub fn new(
        kind: MessageKind,
        ptype: u8,
        validity: MessageValidity,
        length: usize,
        nbuffers: u32,
        nchildren: u32,
    ) -> Self {
        let mut out = Vec::with_capacity(MESSAGE_HEADER_LEN);
        out.push(kind as u8);
        out.push(ptype);
        out.push(validity as u8);
        out.push(0);
        out.extend_from_slice(&(length as u64).to_le_bytes());
        out.extend_from_slice(&nbuffers.to_le_bytes());
        out.extend_from_slice(&nchildren.to_le_bytes());
        Self { out }
    }

    /// Append a buffer (header + inline bytes).
    pub fn buffer(mut self, alignment_exponent: u8, bytes: &[u8]) -> Self {
        self.out
            .extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        self.out.push(alignment_exponent);
        self.out.extend_from_slice(&[0u8; 7]);
        self.out.extend_from_slice(bytes);
        self
    }

    /// Append an already-encoded child message.
    pub fn child(mut self, child: &[u8]) -> Self {
        self.out.extend_from_slice(child);
        self
    }

    /// Finish, returning the message bytes.
    pub fn finish(self) -> Vec<u8> {
        self.out
    }
}

/// Convenience: build a non-nullable primitive message from raw little-endian bytes.
pub fn primitive_message(ptype: PType, length: usize, data: &[u8]) -> Vec<u8> {
    let alignment_exponent = ptype.byte_width().trailing_zeros() as u8;
    MessageBuilder::new(
        MessageKind::Primitive,
        ptype as u8,
        MessageValidity::NonNullable,
        length,
        1,
        0,
    )
    .buffer(alignment_exponent, data)
    .finish()
}

/// Convenience: build a null message.
pub fn null_message(length: usize) -> Vec<u8> {
    MessageBuilder::new(
        MessageKind::Null,
        0,
        MessageValidity::NonNullable,
        length,
        0,
        0,
    )
    .finish()
}

/// A read-only view over a received [`CanonicalMessage`] (e.g. a host-decoded child).
pub struct MessageReader<'a> {
    bytes: &'a [u8],
}

impl<'a> MessageReader<'a> {
    /// Wrap a message byte blob.
    pub fn new(bytes: &'a [u8]) -> VortexResult<Self> {
        vortex_ensure!(
            bytes.len() >= MESSAGE_HEADER_LEN,
            "message shorter than header"
        );
        Ok(Self { bytes })
    }

    /// The raw `kind` discriminant.
    pub fn kind(&self) -> u8 {
        self.bytes[0]
    }

    /// The raw `ptype` discriminant.
    pub fn ptype(&self) -> u8 {
        self.bytes[1]
    }

    /// The raw `validity` discriminant.
    pub fn validity(&self) -> u8 {
        self.bytes[2]
    }

    /// The logical element count.
    pub fn length(&self) -> usize {
        u64::from_le_bytes(self.bytes[4..12].try_into().expect("8 bytes")) as usize
    }

    /// The number of buffers.
    pub fn nbuffers(&self) -> u32 {
        u32::from_le_bytes(self.bytes[12..16].try_into().expect("4 bytes"))
    }

    /// The inline bytes of buffer index 0, if present.
    pub fn first_buffer(&self) -> VortexResult<&'a [u8]> {
        vortex_ensure!(self.nbuffers() >= 1, "message has no buffers");
        let entry = MESSAGE_HEADER_LEN;
        let len =
            u64::from_le_bytes(self.bytes[entry..entry + 8].try_into().expect("8 bytes")) as usize;
        let start = entry + BUFFER_ENTRY_HEADER_LEN;
        let end = start + len;
        vortex_ensure!(end <= self.bytes.len(), "message truncated reading buffer");
        Ok(&self.bytes[start..end])
    }

    /// Interpret the first buffer as a slice of little-endian `u32`s.
    pub fn first_buffer_as_u32(&self) -> VortexResult<Vec<u32>> {
        let bytes = self.first_buffer()?;
        vortex_ensure!(bytes.len() % 4 == 0, "buffer not u32-aligned in length");
        Ok(bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().expect("4 bytes")))
            .collect())
    }

    /// Validate that the message describes the expected kind.
    pub fn expect_kind(&self, kind: MessageKind) -> VortexResult<()> {
        if self.kind() != kind as u8 {
            vortex_bail!("expected message kind {:?}, got {}", kind, self.kind());
        }
        Ok(())
    }
}
