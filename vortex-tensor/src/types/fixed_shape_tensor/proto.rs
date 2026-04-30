// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Protobuf serialization for [`FixedShapeTensorMetadata`].

use prost::Message;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::types::fixed_shape_tensor::FixedShapeTensorMetadata;

/// Protobuf representation of [`FixedShapeTensorMetadata`].
///
/// Protobuf does not distinguish between an absent repeated field and an empty one (both will
/// deserialize as an empty `Vec`). This is fine because the semantic meaning is unambiguous:
///
/// - `logical_shape` empty: 0-dimensional (scalar) tensor.
/// - `dim_names` empty: no dimension names (`None`).
/// - `permutation` empty: no permutation, i.e., identity layout (`None`).
#[derive(Clone, PartialEq, Message)]
struct FixedShapeTensorMetadataProto {
    /// The size of each logical dimension. Empty for a 0-dimensional scalar tensor.
    #[prost(uint32, repeated, tag = "1")]
    logical_shape: Vec<u32>,

    /// Optional human-readable names for each logical dimension. When present, must have the
    /// same length as `logical_shape`. Empty means no names are set.
    #[prost(string, repeated, tag = "2")]
    dim_names: Vec<String>,

    /// Optional dimension permutation mapping logical to physical indices. When present, must
    /// be a permutation of `[0, 1, ..., N-1]`. Empty means identity (row-major) layout.
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
///
/// For 0-dimensional tensors, all three repeated fields are empty, which correctly produces a
/// metadata with an empty shape and no names or permutation.
pub(crate) fn deserialize(bytes: &[u8]) -> VortexResult<FixedShapeTensorMetadata> {
    let proto = FixedShapeTensorMetadataProto::decode(bytes).map_err(|e| vortex_err!("{e}"))?;

    let logical_shape = proto
        .logical_shape
        .into_iter()
        .map(|d| d as usize)
        .collect();
    let mut m = FixedShapeTensorMetadata::new(logical_shape);

    // Note that this is fine for 0 dimensions since if we do not have any dimensions, we cannot
    // have any names or permutations.
    if !proto.dim_names.is_empty() {
        m = m.with_dim_names(proto.dim_names)?;
    }
    if !proto.permutation.is_empty() {
        let permutation = proto.permutation.into_iter().map(|i| i as usize).collect();
        m = m.with_permutation(permutation)?;
    }

    Ok(m)
}
