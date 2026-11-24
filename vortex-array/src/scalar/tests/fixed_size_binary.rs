// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for fixed-size binary scalars.

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use rstest::rstest;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::dtype::DType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::Nullability::Nullable;
    use crate::proto::scalar as pb;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;

    fn fsb_dtype(size: u32) -> DType {
        DType::FixedSizeBinary(size, NonNullable)
    }

    #[test]
    fn constructor_infers_width() {
        let scalar = Scalar::fixed_size_binary(vec![1u8, 2, 3], Nullable);
        assert_eq!(scalar.dtype(), &DType::FixedSizeBinary(3, Nullable));
        assert_eq!(
            scalar.as_binary().value().map(ByteBuffer::as_slice),
            Some([1u8, 2, 3].as_slice())
        );
    }

    #[rstest]
    #[case::too_short(vec![1u8, 2])]
    #[case::too_long(vec![1u8, 2, 3, 4])]
    fn rejects_wrong_length(#[case] bytes: Vec<u8>) {
        let result = Scalar::try_new(fsb_dtype(3), Some(ScalarValue::Binary(bytes.into())));
        assert!(result.is_err());
    }

    #[test]
    fn rejects_non_binary_value() {
        let result = Scalar::try_new(fsb_dtype(1), Some(ScalarValue::Bool(true)));
        assert!(result.is_err());
    }

    #[test]
    fn zero_and_default_are_zeroed_bytes() {
        let zero = Scalar::zero_value(&fsb_dtype(4));
        assert_eq!(zero, Scalar::fixed_size_binary(vec![0u8; 4], NonNullable));
        assert_eq!(zero.is_zero(), Some(true));

        let default = Scalar::default_value(&fsb_dtype(4));
        assert_eq!(default, zero);

        assert!(Scalar::default_value(&fsb_dtype(4).as_nullable()).is_null());
    }

    #[test]
    fn is_zero_requires_every_byte_zero() {
        let scalar = Scalar::fixed_size_binary(vec![0u8, 0, 1], NonNullable);
        assert_eq!(scalar.is_zero(), Some(false));
        assert_eq!(
            Scalar::fixed_size_binary(Vec::<u8>::new(), NonNullable).is_zero(),
            Some(true)
        );
    }

    #[test]
    fn approx_nbytes_is_width() {
        assert_eq!(
            Scalar::fixed_size_binary(vec![7u8; 16], NonNullable).approx_nbytes(),
            16
        );
        assert_eq!(
            Scalar::null(DType::FixedSizeBinary(16, Nullable)).approx_nbytes(),
            16
        );
    }

    #[test]
    fn display() {
        let scalar = Scalar::fixed_size_binary(vec![0x0au8, 0xff], NonNullable);
        assert_eq!(
            scalar.to_string(),
            Scalar::binary(vec![0x0au8, 0xff], NonNullable).to_string()
        );
        assert_eq!(
            Scalar::null(DType::FixedSizeBinary(2, Nullable)).to_string(),
            "null"
        );
    }

    #[test]
    fn ordering_is_bytewise() {
        let a = Scalar::fixed_size_binary(vec![1u8, 2], NonNullable);
        let b = Scalar::fixed_size_binary(vec![1u8, 3], NonNullable);
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Less));
        assert_eq!(
            a.partial_cmp(&Scalar::fixed_size_binary(vec![1u8, 2, 3], NonNullable)),
            None
        );
        assert_eq!(
            a.partial_cmp(&Scalar::binary(vec![1u8, 2], NonNullable)),
            None
        );
    }

    #[test]
    fn cast_to_binary() -> VortexResult<()> {
        let scalar = Scalar::fixed_size_binary(vec![1u8, 2, 3], NonNullable);
        let cast = scalar.cast(&DType::Binary(Nullable))?;
        assert_eq!(cast, Scalar::binary(vec![1u8, 2, 3], Nullable));
        Ok(())
    }

    #[test]
    fn cast_from_binary_with_matching_length() -> VortexResult<()> {
        let scalar = Scalar::binary(vec![1u8, 2, 3], NonNullable);
        let cast = scalar.cast(&fsb_dtype(3))?;
        assert_eq!(
            cast,
            Scalar::fixed_size_binary(vec![1u8, 2, 3], NonNullable)
        );
        Ok(())
    }

    #[rstest]
    #[case::binary_wrong_length(Scalar::binary(vec![1u8, 2], NonNullable), fsb_dtype(3))]
    #[case::other_width(Scalar::fixed_size_binary(vec![1u8, 2], NonNullable), fsb_dtype(3))]
    #[case::utf8(
        Scalar::fixed_size_binary(vec![b'a'], NonNullable),
        DType::Utf8(NonNullable)
    )]
    fn cast_rejects(#[case] scalar: Scalar, #[case] target: DType) {
        assert!(scalar.cast(&target).is_err());
    }

    #[test]
    fn cast_null() -> VortexResult<()> {
        let scalar = Scalar::null(DType::FixedSizeBinary(2, Nullable));
        assert!(scalar.cast(&DType::Binary(Nullable))?.is_null());
        assert!(scalar.cast(&fsb_dtype(2)).is_err());
        Ok(())
    }

    #[rstest]
    #[case::value(Scalar::fixed_size_binary(vec![9u8, 8, 7, 6], Nullable))]
    #[case::null(Scalar::null(DType::FixedSizeBinary(4, Nullable)))]
    fn proto_round_trip(#[case] scalar: Scalar) -> VortexResult<()> {
        let pb = pb::Scalar::from(&scalar);
        let read_back = Scalar::from_proto(&pb, &VortexSession::empty())?;
        assert_eq!(read_back, scalar);
        assert_eq!(read_back.dtype(), scalar.dtype());
        Ok(())
    }
}
