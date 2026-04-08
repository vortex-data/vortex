// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Protobuf-backed metadata for TurboQuant encoding.

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::encodings::turboquant::TurboQuant;

/// Serialized metadata for TurboQuant arrays.
#[derive(Clone, PartialEq, Message)]
pub(super) struct TurboQuantMetadata {
    /// The number of bits per coordinate, which must be <= [`TurboQuant::MAX_BIT_WIDTH`].
    #[prost(uint32, required, tag = "1")]
    bit_width: u32,

    /// The number of sign-diagonal + WHT rounds in the structured rotation.
    #[prost(uint32, required, tag = "2")]
    num_rounds: u32,
}

impl TurboQuantMetadata {
    /// Creates metadata for the given bit width and number of rotation rounds.
    pub(super) fn new(bit_width: u8, num_rounds: u8) -> Self {
        Self {
            bit_width: u32::from(bit_width),
            num_rounds: u32::from(num_rounds),
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
            bit_width <= TurboQuant::MAX_BIT_WIDTH,
            "bit_width is expected to be between 0 and {}, got {bit_width}",
            TurboQuant::MAX_BIT_WIDTH
        );

        Ok(bit_width)
    }

    /// Returns the validated number of rotation rounds.
    ///
    /// Returns 0 for degenerate (empty) arrays, which is validated at a higher level.
    pub(super) fn num_rounds(&self) -> VortexResult<u8> {
        u8::try_from(self.num_rounds).map_err(|_| {
            vortex_err!(
                "TurboQuant num_rounds must fit into u8, got {}",
                self.num_rounds
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use prost::Message;
    use rstest::rstest;
    use vortex_error::VortexResult;

    use super::TurboQuantMetadata;

    #[rstest]
    #[case(0, 0)]
    #[case(0, 3)]
    #[case(3, 1)]
    #[case(8, 3)]
    #[case(8, 5)]
    fn protobuf_metadata_roundtrip(
        #[case] bit_width: u8,
        #[case] num_rounds: u8,
    ) -> VortexResult<()> {
        let bytes = TurboQuantMetadata::new(bit_width, num_rounds).encode_to_vec();
        let decoded = TurboQuantMetadata::decode(bytes.as_slice())?;
        assert_eq!(decoded.bit_width()?, bit_width);
        assert_eq!(decoded.num_rounds()?, num_rounds);

        Ok(())
    }
}
