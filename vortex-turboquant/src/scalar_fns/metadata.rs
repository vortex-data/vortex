// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::TurboQuantConfig;

#[derive(Clone, PartialEq, Message)]
pub(super) struct TQScalarFnMetadata {
    #[prost(uint32, tag = "1")]
    bit_width: u32,
    #[prost(uint64, tag = "2")]
    seed: u64,
    #[prost(uint32, tag = "3")]
    num_rounds: u32,
    /// Optional user-supplied block decomposition. An empty repeated field on the wire (default)
    /// decodes to `block_sizes: None`, and one or more entries decode to `Some(vec![..])`.
    #[prost(uint32, repeated, tag = "4")]
    block_sizes: Vec<u32>,
}

impl TQScalarFnMetadata {
    pub(super) fn from_config(config: &TurboQuantConfig) -> Self {
        Self {
            bit_width: config.bit_width() as u32,
            seed: config.seed(),
            num_rounds: config.num_rounds() as u32,
            block_sizes: config
                .block_sizes()
                .map(<[u32]>::to_vec)
                .unwrap_or_default(),
        }
    }

    pub(super) fn to_config(&self) -> VortexResult<TurboQuantConfig> {
        let bit_width = u8::try_from(self.bit_width)
            .map_err(|_| vortex_err!("TurboQuant bit_width does not fit u8"))?;
        let num_rounds = u8::try_from(self.num_rounds)
            .map_err(|_| vortex_err!("TurboQuant num_rounds does not fit u8"))?;
        let block_sizes = if self.block_sizes.is_empty() {
            None
        } else {
            Some(self.block_sizes.clone())
        };

        TurboQuantConfig::try_new(bit_width, self.seed, num_rounds, block_sizes)
    }
}

pub(super) fn serialize_config(config: &TurboQuantConfig) -> Vec<u8> {
    TQScalarFnMetadata::from_config(config).encode_to_vec()
}

pub(super) fn deserialize_config(metadata: &[u8]) -> VortexResult<TurboQuantConfig> {
    TQScalarFnMetadata::decode(metadata)
        .map_err(|e| vortex_err!("Failed to decode TurboQuant scalar function metadata: {e}"))?
        .to_config()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_roundtrips_block_sizes_none() -> VortexResult<()> {
        let config = TurboQuantConfig::try_new(3, 7, 2, None)?;
        let bytes = serialize_config(&config);
        let round = deserialize_config(&bytes)?;
        assert_eq!(round.block_sizes(), None);
        assert_eq!(round, config);
        Ok(())
    }

    #[test]
    fn serialize_roundtrips_block_sizes_some() -> VortexResult<()> {
        let config = TurboQuantConfig::try_new(3, 7, 2, Some(vec![512, 256]))?;
        let bytes = serialize_config(&config);
        let round = deserialize_config(&bytes)?;
        assert_eq!(round.block_sizes(), Some([512, 256].as_slice()));
        assert_eq!(round, config);
        Ok(())
    }
}
