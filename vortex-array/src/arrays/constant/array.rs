// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_scalar::Scalar;

use crate::stats::ArrayStats;

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
    use rstest::rstest;
    use vortex_buffer::ByteBufferMut;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;
    use vortex_error::VortexResult;
    use vortex_scalar::Scalar;
    use vortex_scalar::ScalarValue;
    use vortex_session::VortexSession;

    use crate::arrays::ConstantArray;
    use crate::arrays::constant::vtable::CONSTANT_INLINE_THRESHOLD;
    use crate::arrays::constant::vtable::ConstantVTable;
    use crate::buffer::BufferHandle;
    use crate::vtable::VTable;

    #[rstest]
    #[case::below_threshold(CONSTANT_INLINE_THRESHOLD - 1, true)]
    #[case::at_threshold(CONSTANT_INLINE_THRESHOLD, true)]
    #[case::above_threshold(CONSTANT_INLINE_THRESHOLD + 1, false)]
    fn test_serialize_inlining(
        #[case] nbytes: usize,
        #[case] should_inline: bool,
    ) -> VortexResult<()> {
        // UTF-8 scalar `nbytes` equals the string length.
        let string = "x".repeat(nbytes);
        let array = ConstantArray::new(Scalar::from(string.as_str()), 10);
        let metadata = ConstantVTable::metadata(&array)?;
        let serialized =
            ConstantVTable::serialize(metadata)?.expect("serialize should produce Some bytes");

        assert_eq!(
            !serialized.is_empty(),
            should_inline,
            "scalar of {nbytes} bytes: expected inlined={should_inline}"
        );
        Ok(())
    }

    #[test]
    fn test_metadata_round_trips() -> VortexResult<()> {
        let scalar = Scalar::from(42i64);
        let array = ConstantArray::new(scalar.clone(), 5);
        let metadata = ConstantVTable::metadata(&array)?;

        // Serialize and deserialize the metadata.
        let bytes =
            ConstantVTable::serialize(metadata)?.expect("serialize should produce Some bytes");
        let session = VortexSession::empty();
        let deserialized = ConstantVTable::deserialize(
            &bytes,
            &DType::Primitive(PType::I64, Nullability::NonNullable),
            5,
            &[],
            &session,
        )?;

        assert_eq!(deserialized, scalar);
        Ok(())
    }

    #[test]
    fn test_empty_bytes_deserializes_from_buffer() -> VortexResult<()> {
        let scalar_value = ScalarValue::from(42i32);
        let buffer = ScalarValue::to_proto_bytes::<ByteBufferMut>(Some(&scalar_value)).freeze();
        let buffer_handle = BufferHandle::new_host(buffer);
        let session = VortexSession::empty();
        let metadata = ConstantVTable::deserialize(
            &[],
            &DType::Primitive(PType::I32, Nullability::NonNullable),
            10,
            &[buffer_handle],
            &session,
        )?;
        assert_eq!(metadata, Scalar::from(42i32));
        Ok(())
    }
}
