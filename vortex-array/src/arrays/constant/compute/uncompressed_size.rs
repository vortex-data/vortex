// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(test)]
mod tests {
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use crate::ArrayRef;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
    use crate::aggregate_fn::fns::uncompressed_size_in_bytes::canonical_uncompressed_size_in_bytes;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    fn aggregate(array: &ArrayRef) -> VortexResult<u64> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let mut acc =
            Accumulator::try_new(UncompressedSizeInBytes, EmptyOptions, array.dtype().clone())?;
        acc.accumulate(array, &mut ctx)?;
        acc.finish()?
            .as_primitive()
            .typed_value::<u64>()
            .ok_or_else(|| vortex_err!("uncompressed size result should not be null"))
    }

    #[test]
    fn constant_primitive_matches_primitive_array() -> VortexResult<()> {
        let constant = ConstantArray::new(5i32, 10).into_array();
        let primitive =
            PrimitiveArray::new(Buffer::full(5i32, 10), Validity::NonNullable).into_array();

        assert_eq!(aggregate(&constant)?, aggregate(&primitive)?);
        Ok(())
    }

    #[test]
    fn nullable_constant_primitive_matches_nullable_primitive_array() -> VortexResult<()> {
        let constant =
            ConstantArray::new(Scalar::primitive(5i32, Nullability::Nullable), 10).into_array();
        let primitive =
            PrimitiveArray::new(Buffer::full(5i32, 10), Validity::AllValid).into_array();

        assert_eq!(aggregate(&constant)?, aggregate(&primitive)?);
        Ok(())
    }

    #[test]
    fn null_constant_primitive_matches_all_invalid_primitive_array() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let constant = ConstantArray::new(Scalar::null(dtype), 10).into_array();
        let primitive =
            PrimitiveArray::new(Buffer::<i32>::zeroed(10), Validity::AllInvalid).into_array();

        assert_eq!(aggregate(&constant)?, aggregate(&primitive)?);
        Ok(())
    }

    #[test]
    fn constant_bool_uses_packed_canonical_size() -> VortexResult<()> {
        let constant = ConstantArray::new(true, 10).into_array();
        let bool_array = BoolArray::from_iter([true; 10]).into_array();

        assert_eq!(aggregate(&constant)?, aggregate(&bool_array)?);
        assert_eq!(aggregate(&constant)?, 2);
        Ok(())
    }

    #[test]
    fn constant_utf8_matches_canonical_size() -> VortexResult<()> {
        let constant = ConstantArray::new("abcdefghijkl".to_string(), 10).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let canonical = constant.clone().execute::<Canonical>(&mut ctx)?;
        let expected = canonical_uncompressed_size_in_bytes(&canonical, &mut ctx)?;

        assert_eq!(aggregate(&constant)?, expected);
        Ok(())
    }
}
