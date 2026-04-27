// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! On-disk protobuf serialization for [`FixedShapeTensorMetadata`].
//!
//! The Arrow JSON wire is a separate concern; see [`super::canonical`] for the proto↔JSON
//! adapters invoked at the Arrow boundary.

use prost::Message;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::types::fixed_shape::FixedShapeTensorMetadata;

/// Empty repeated fields collapse with absent ones in proto, which matches our semantics:
/// empty `logical_shape` is a scalar; empty `dim_names`/`permutation` mean `None`.
#[derive(Clone, PartialEq, Message)]
struct FixedShapeTensorMetadataProto {
    #[prost(uint32, repeated, tag = "1")]
    logical_shape: Vec<u32>,
    #[prost(string, repeated, tag = "2")]
    dim_names: Vec<String>,
    #[prost(uint32, repeated, tag = "3")]
    permutation: Vec<u32>,
}

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

pub(crate) fn deserialize(bytes: &[u8]) -> VortexResult<FixedShapeTensorMetadata> {
    let proto = FixedShapeTensorMetadataProto::decode(bytes).map_err(|e| vortex_err!("{e}"))?;
    let logical_shape = proto
        .logical_shape
        .into_iter()
        .map(|d| d as usize)
        .collect();
    let mut m = FixedShapeTensorMetadata::new(logical_shape);
    if !proto.dim_names.is_empty() {
        m = m.with_dim_names(proto.dim_names)?;
    }
    if !proto.permutation.is_empty() {
        let permutation = proto.permutation.into_iter().map(|i| i as usize).collect();
        m = m.with_permutation(permutation)?;
    }
    Ok(m)
}
