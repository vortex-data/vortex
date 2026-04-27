// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow canonical [`arrow.fixed_shape_tensor`] JSON wire ⇄ on-disk proto adapters.
//!
//! Hand-rolled rather than reusing `arrow_schema::extension::FixedShapeTensor` because arrow-rs
//! 58 emits `"permutations"` (plural) while the spec and pyarrow use `"permutation"`.
//!
//! [`arrow.fixed_shape_tensor`]: https://arrow.apache.org/docs/format/CanonicalExtensions.html#fixed-shape-tensor

use serde::Deserialize;
use serde::Serialize;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::types::fixed_shape::FixedShapeTensorMetadata;
use crate::types::fixed_shape::proto;

#[derive(Serialize)]
struct WireRef<'a> {
    shape: &'a [usize],
    #[serde(skip_serializing_if = "Option::is_none")]
    dim_names: Option<&'a [String]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    permutation: Option<&'a [usize]>,
}

#[derive(Deserialize)]
struct Wire {
    shape: Vec<usize>,
    #[serde(default)]
    dim_names: Option<Vec<String>>,
    #[serde(default)]
    permutation: Option<Vec<usize>>,
}

fn metadata_to_json(metadata: &FixedShapeTensorMetadata) -> VortexResult<String> {
    let wire = WireRef {
        shape: metadata.logical_shape(),
        dim_names: metadata.dim_names(),
        permutation: metadata.permutation(),
    };
    serde_json::to_string(&wire)
        .map_err(|e| vortex_err!("fixed_shape_tensor canonical serialize: {e}"))
}

fn metadata_from_json(json: &str) -> VortexResult<FixedShapeTensorMetadata> {
    let wire: Wire = serde_json::from_str(json)
        .map_err(|e| vortex_err!("fixed_shape_tensor canonical deserialize: {e}"))?;

    let mut m = FixedShapeTensorMetadata::new(wire.shape);
    if let Some(names) = wire.dim_names {
        m = m.with_dim_names(names)?;
    }
    if let Some(perm) = wire.permutation {
        m = m.with_permutation(perm)?;
    }
    Ok(m)
}

pub(crate) fn proto_to_json(proto_bytes: &[u8]) -> VortexResult<String> {
    let metadata = proto::deserialize(proto_bytes)?;
    metadata_to_json(&metadata)
}

pub(crate) fn json_to_proto(json: &str) -> VortexResult<Vec<u8>> {
    let metadata = metadata_from_json(json)?;
    Ok(proto::serialize(&metadata))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::scalar_0d(FixedShapeTensorMetadata::new(vec![]))]
    #[case::vector_1d(FixedShapeTensorMetadata::new(vec![5]))]
    #[case::shape_only(FixedShapeTensorMetadata::new(vec![2, 3, 4]))]
    #[case::with_dim_names(
        FixedShapeTensorMetadata::new(vec![3, 4])
            .with_dim_names(vec!["rows".into(), "cols".into()])
            .unwrap()
    )]
    #[case::with_permutation(
        FixedShapeTensorMetadata::new(vec![2, 3, 4])
            .with_permutation(vec![2, 0, 1])
            .unwrap()
    )]
    #[case::all_fields(
        FixedShapeTensorMetadata::new(vec![2, 3, 4])
            .with_dim_names(vec!["x".into(), "y".into(), "z".into()]).unwrap()
            .with_permutation(vec![1, 2, 0]).unwrap()
    )]
    fn json_roundtrip(#[case] metadata: FixedShapeTensorMetadata) -> VortexResult<()> {
        let json = metadata_to_json(&metadata)?;
        let decoded = metadata_from_json(&json)?;
        assert_eq!(decoded, metadata);
        Ok(())
    }

    #[rstest]
    #[case::shape_only(FixedShapeTensorMetadata::new(vec![2, 3, 4]))]
    #[case::all_fields(
        FixedShapeTensorMetadata::new(vec![2, 3, 4])
            .with_dim_names(vec!["x".into(), "y".into(), "z".into()]).unwrap()
            .with_permutation(vec![1, 2, 0]).unwrap()
    )]
    fn proto_to_json_to_proto_roundtrip(
        #[case] metadata: FixedShapeTensorMetadata,
    ) -> VortexResult<()> {
        let proto_bytes = proto::serialize(&metadata);
        let json = proto_to_json(&proto_bytes)?;
        let proto_again = json_to_proto(&json)?;
        let metadata_again = proto::deserialize(&proto_again)?;
        assert_eq!(metadata_again, metadata);
        Ok(())
    }

    #[test]
    fn wire_format_matches_arrow_spec() -> VortexResult<()> {
        let metadata = FixedShapeTensorMetadata::new(vec![2, 3, 4])
            .with_dim_names(vec!["x".into(), "y".into(), "z".into()])?
            .with_permutation(vec![1, 2, 0])?;

        let json = metadata_to_json(&metadata)?;
        let v: serde_json::Value =
            serde_json::from_str(&json).map_err(|e| vortex_err!("parse wire: {e}"))?;

        assert_eq!(v["shape"], serde_json::json!([2, 3, 4]));
        assert_eq!(v["dim_names"], serde_json::json!(["x", "y", "z"]));
        // Arrow spec uses singular "permutation"; guard against regressions to arrow-rs's plural.
        assert_eq!(v["permutation"], serde_json::json!([1, 2, 0]));
        assert!(v.get("permutations").is_none());
        Ok(())
    }

    #[test]
    fn omits_optional_fields_when_unset() -> VortexResult<()> {
        let json = metadata_to_json(&FixedShapeTensorMetadata::new(vec![5]))?;
        let v: serde_json::Value =
            serde_json::from_str(&json).map_err(|e| vortex_err!("parse wire: {e}"))?;
        assert!(v.get("dim_names").is_none());
        assert!(v.get("permutation").is_none());
        Ok(())
    }
}
