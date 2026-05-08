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
}

impl TQScalarFnMetadata {
    pub(super) fn from_config(config: &TurboQuantConfig) -> Self {
        Self {
            bit_width: u32::from(config.bit_width()),
            seed: config.seed(),
            num_rounds: u32::from(config.num_rounds()),
        }
    }

    pub(super) fn to_config(&self) -> VortexResult<TurboQuantConfig> {
        let bit_width = u8::try_from(self.bit_width)
            .map_err(|_| vortex_err!("TurboQuant bit_width does not fit u8"))?;
        let num_rounds = u8::try_from(self.num_rounds)
            .map_err(|_| vortex_err!("TurboQuant num_rounds does not fit u8"))?;

        TurboQuantConfig::try_new(bit_width, self.seed, num_rounds)
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
