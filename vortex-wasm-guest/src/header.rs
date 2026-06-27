// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Parsing the Vortex array flatbuffer header from inside a guest kernel, using only
//! `vortex-flatbuffers` — never the rest of Vortex.
//!
//! The serialized array is laid out as `[data buffers][flatbuffer][u32 LE flatbuffer length]`.
//! This module locates and parses the flatbuffer so a kernel can read its own encoding metadata,
//! buffer table, and child nodes exactly as a native `VTable::deserialize` would.

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_flatbuffers::array as fba;

/// A parsed view over a serialized Vortex array's flatbuffer header.
pub struct ArrayHeader<'a> {
    array: fba::Array<'a>,
}

impl<'a> ArrayHeader<'a> {
    /// Parse the flatbuffer header out of a serialized array `input`.
    pub fn parse(input: &'a [u8]) -> VortexResult<Self> {
        vortex_ensure!(
            input.len() >= 4,
            "serialized array shorter than length suffix"
        );
        let fb_len = u32::from_le_bytes(
            input[input.len() - 4..]
                .try_into()
                .expect("4 byte length suffix"),
        ) as usize;
        vortex_ensure!(
            input.len() >= 4 + fb_len,
            "serialized array shorter than declared flatbuffer"
        );
        let fb_start = input.len() - 4 - fb_len;
        let fb = &input[fb_start..fb_start + fb_len];
        let array = fba::root_as_array(fb)
            .map_err(|e| vortex_error::vortex_err!("invalid array flatbuffer: {e}"))?;
        Ok(Self { array })
    }

    /// The root encoding node.
    pub fn root(&self) -> VortexResult<fba::ArrayNode<'a>> {
        self.array
            .root()
            .ok_or_else(|| vortex_error::vortex_err!("array flatbuffer missing root node"))
    }

    /// The interned encoding index of the root node.
    pub fn encoding(&self) -> VortexResult<u16> {
        Ok(self.root()?.encoding())
    }

    /// The encoding metadata bytes of the root node (empty if none).
    pub fn metadata(&self) -> VortexResult<&'a [u8]> {
        Ok(self.root()?.metadata().map(|m| m.bytes()).unwrap_or(&[]))
    }

    /// The number of child nodes of the root node.
    ///
    /// A child at index `i` can be decoded by the host via
    /// [`crate::host::decode_child(i)`](crate::host::decode_child).
    pub fn nchildren(&self) -> VortexResult<usize> {
        Ok(self.root()?.children().map(|c| c.len()).unwrap_or(0))
    }
}
