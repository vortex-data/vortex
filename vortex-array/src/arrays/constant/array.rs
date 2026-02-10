// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_scalar::Scalar;

use crate::stats::ArrayStats;

/// Protobuf-encoded metadata for [`ConstantArray`].
///
/// When the serialized scalar value is small enough (see `CONSTANT_INLINE_THRESHOLD`),
/// it is inlined directly in the metadata to avoid a device-to-host copy on GPU.
#[derive(Clone, prost::Message)]
pub struct ConstantMetadata {
    #[prost(optional, bytes, tag = "1")]
    pub(super) scalar_value: Option<Vec<u8>>,
}

#[derive(Clone, Debug)]
pub struct ConstantArray {
    pub(super) scalar: Scalar,
    pub(super) len: usize,
    pub(super) stats_set: ArrayStats,
}

impl ConstantArray {
    pub fn new<S>(scalar: S, len: usize) -> Self
    where
        S: Into<Scalar>,
    {
        let scalar = scalar.into();
        Self {
            scalar,
            len,
            stats_set: Default::default(),
        }
    }

    /// Returns the [`Scalar`] value of this constant array.
    pub fn scalar(&self) -> &Scalar {
        &self.scalar
    }

    pub fn into_parts(self) -> Scalar {
        self.scalar
    }
}

#[cfg(test)]
mod tests {
    use vortex_scalar::ScalarValue;

    use super::ConstantMetadata;
    use crate::ProstMetadata;
    use crate::test_harness::check_metadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_constant_metadata() {
        let scalar_bytes: Vec<u8> = ScalarValue::from(i32::MAX).to_protobytes();
        check_metadata(
            "constant.metadata",
            ProstMetadata(ConstantMetadata {
                scalar_value: Some(scalar_bytes),
            }),
        );
    }
}
