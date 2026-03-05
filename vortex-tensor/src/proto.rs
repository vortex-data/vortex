// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Protobuf serialization for [`FixedShapeTensorMetadata`].

use prost::Message;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

use crate::FixedShapeTensorMetadata;

/// Protobuf representation of [`FixedShapeTensorMetadata`].
#[derive(Clone, PartialEq, Message)]
struct FixedShapeTensorMetadataProto {
    #[prost(uint32, repeated, tag = "1")]
    logical_shape: Vec<u32>,
    #[prost(string, repeated, tag = "2")]
    dim_names: Vec<String>,
    #[prost(uint32, repeated, tag = "3")]
    permutation: Vec<u32>,
}

/// Serializes [`FixedShapeTensorMetadata`] to protobuf bytes.
pub(crate) fn serialize(metadata: &FixedShapeTensorMetadata) -> Vec<u8> {
    let logical_shape = metadata
        .logical_shape()
        .iter()
        .map(|&d| u32::try_from(d).vortex_expect("dimension size exceeds u32"))
        .collect();

    let dim_names = metadata.dim_names().map(|n| n.to_vec()).unwrap_or_default();

    let permutation = metadata
        .permutation()
        .map(|p| {
            p.iter()
                .map(|&i| u32::try_from(i).vortex_expect("permutation index exceeds u32"))
                .collect()
        })
        .unwrap_or_default();

    let proto = FixedShapeTensorMetadataProto {
        logical_shape,
        dim_names,
        permutation,
    };
    proto.encode_to_vec()
}

/// Deserializes [`FixedShapeTensorMetadata`] from protobuf bytes.
pub(crate) fn deserialize(bytes: &[u8]) -> VortexResult<FixedShapeTensorMetadata> {
    let proto = FixedShapeTensorMetadataProto::decode(bytes).map_err(|e| vortex_err!("{e}"))?;

    let logical_shape = proto
        .logical_shape
        .into_iter()
        .map(|d| d as usize)
        .collect();
    let mut m = FixedShapeTensorMetadata::new(logical_shape);

    if !proto.dim_names.is_empty() {
        m = m.with_dim_names(proto.dim_names);
    }
    if !proto.permutation.is_empty() {
        let permutation = proto.permutation.into_iter().map(|i| i as usize).collect();
        m = m.with_permutation(permutation);
    }

    Ok(m)
}
