// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Protobuf-backed metadata for TurboQuant encoding.

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

/// Serialized metadata for TurboQuant arrays.
#[derive(Clone, PartialEq, Message)]
pub(super) struct TurboQuantMetadata {
    /// The number of bits per coordinate.
    #[prost(uint32, required, tag = "1")]
    bit_width: u32,
}

impl TurboQuantMetadata {
    /// Creates metadata for the given bit width.
    pub(super) fn new(bit_width: u8) -> Self {
        Self {
            bit_width: u32::from(bit_width),
        }
    }

    /// Returns the validated TurboQuant bit width.
    pub(super) fn bit_width(&self) -> VortexResult<u8> {
        let bit_width = u8::try_from(self.bit_width).map_err(|_| {
            vortex_err!(
                "TurboQuant bit_width must fit into u8, got {}",
                self.bit_width
            )
        })?;
        vortex_ensure!(
            bit_width <= 8,
            "bit_width is expected to be between 0 and 8, got {bit_width}"
        );

        Ok(bit_width)
    }
}

#[cfg(test)]
mod tests {
    use prost::Message;
    use rstest::rstest;
    use vortex_error::VortexResult;

    use super::TurboQuantMetadata;

    #[rstest]
    #[case(0)]
    #[case(3)]
    #[case(8)]
    fn protobuf_metadata_roundtrip(#[case] bit_width: u8) -> VortexResult<()> {
        let bytes = TurboQuantMetadata::new(bit_width).encode_to_vec();
        assert_eq!(
            TurboQuantMetadata::decode(bytes.as_slice())?.bit_width()?,
            bit_width
        );

        Ok(())
    }
}
