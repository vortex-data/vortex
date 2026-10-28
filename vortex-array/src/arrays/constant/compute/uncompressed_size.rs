// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
    use crate::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeOpts;
    use crate::aggregate_fn::fns::uncompressed_size_in_bytes::canonical_uncompressed_size_in_bytes;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn aggregate(array: &ArrayRef) -> VortexResult<u64> {
        aggregate_with(array, UncompressedSizeOpts::default())
    }

    fn aggregate_with(array: &ArrayRef, opts: UncompressedSizeOpts) -> VortexResult<u64> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut acc = Accumulator::try_new(UncompressedSizeInBytes, opts, array.dtype().clone())?;
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
        assert_eq!(aggregate(&constant)?, 40);
        Ok(())
    }

    #[test]
    fn nullable_constant_primitive_matches_nullable_primitive_array() -> VortexResult<()> {
        let constant =
            ConstantArray::new(Scalar::primitive(5i32, Nullability::Nullable), 10).into_array();
        let primitive =
            PrimitiveArray::new(Buffer::full(5i32, 10), Validity::AllValid).into_array();

        assert_eq!(aggregate(&constant)?, aggregate(&primitive)?);
        assert_eq!(aggregate(&constant)?, 40);
        Ok(())
    }

    #[test]
    fn null_constant_primitive_matches_all_invalid_primitive_array() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let constant = ConstantArray::new(Scalar::null(dtype), 10).into_array();
        let primitive =
            PrimitiveArray::new(Buffer::<i32>::zeroed(10), Validity::AllInvalid).into_array();

        assert_eq!(aggregate(&constant)?, aggregate(&primitive)?);
        // 40 value bytes plus the serialized false validity scalar.
        assert_eq!(aggregate(&constant)?, 40 + 2);
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
        let mut ctx = SESSION.create_execution_ctx();
        let canonical = constant.clone().execute::<Canonical>(&mut ctx)?;
        let expected = canonical_uncompressed_size_in_bytes(
            &canonical,
            UncompressedSizeOpts::default(),
            &mut ctx,
        )?;

        assert_eq!(aggregate(&constant)?, expected);
        assert_eq!(aggregate(&constant)?, 160);
        Ok(())
    }

    #[rstest]
    #[case::exact(UncompressedSizeOpts::exact(), 10 * (16 + 20))]
    #[case::inexact(UncompressedSizeOpts::inexact(), 10 * 16 + 20)]
    fn constant_out_of_line_utf8_matches_canonical_size(
        #[case] opts: UncompressedSizeOpts,
        #[case] size: u64,
    ) -> VortexResult<()> {
        // Every row references the one out-of-line value, which canonicalization stores once.
        let constant = ConstantArray::new("abcdefghijklmnopqrst".to_string(), 10).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let canonical = constant.clone().execute::<Canonical>(&mut ctx)?;
        let expected = canonical_uncompressed_size_in_bytes(&canonical, opts, &mut ctx)?;

        assert_eq!(aggregate_with(&constant, opts)?, expected);
        assert_eq!(aggregate_with(&constant, opts)?, size);
        Ok(())
    }

    #[test]
    fn constant_utf8_exact_matches_inexact() -> VortexResult<()> {
        // A constant canonicalizes to a single shared value, so nothing is unreferenced.
        let constant = ConstantArray::new("abcdefghijkl".to_string(), 10).into_array();

        assert_eq!(
            aggregate_with(&constant, UncompressedSizeOpts::exact())?,
            aggregate_with(&constant, UncompressedSizeOpts::inexact())?
        );
        assert_eq!(
            aggregate_with(&constant, UncompressedSizeOpts::exact())?,
            160
        );
        Ok(())
    }
}
