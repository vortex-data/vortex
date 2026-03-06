// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar function to parse UTF-8 strings into [`Uuid`] extension arrays.

use std::fmt::Formatter;
use std::sync::Arc;

use uuid;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::DynArray;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeListArray;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::extension::ExtDType;
use crate::expr::Expression;
use crate::extension::uuid::Uuid;
use crate::extension::uuid::UuidMetadata;
use crate::extension::uuid::vtable::UUID_BYTE_LEN;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

/// Parses a UTF-8 string column into a [`Uuid`] extension array.
///
/// Accepts any standard UUID string format (hyphenated, simple, braced, URN). Invalid strings
/// cause an error.
#[derive(Clone)]
pub struct UuidFromString;

#[expect(
    clippy::cast_possible_truncation,
    reason = "UUID_BYTE_LEN always fits both usize and u32"
)]
impl ScalarFnVTable for UuidFromString {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new_ref("vortex.uuid_from_string")
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("uuid_from_string must have exactly one child"),
        }
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "uuid_from_string(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        debug_assert_eq!(arg_dtypes.len(), 1);

        let input = &arg_dtypes[0];
        vortex_ensure!(
            input.is_utf8(),
            "uuid_from_string requires a Utf8 input, got {input}"
        );

        let nullability = input.nullability();

        let storage_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            UUID_BYTE_LEN as u32,
            nullability,
        );

        let ext_dtype = ExtDType::<Uuid>::try_new(UuidMetadata, storage_dtype)?.erased();

        Ok(DType::Extension(ext_dtype))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let row_count = args.row_count();

        let varbinview = input
            .to_canonical()
            .map_err(|e| vortex_err!("uuid_from_string: failed to canonicalize input: {e}"))?
            .into_varbinview();

        let validity = varbinview.validity()?;

        let mut bytes = vec![0u8; row_count * UUID_BYTE_LEN];

        for i in 0..row_count {
            if !validity.is_valid(i)? {
                continue;
            }

            let str_bytes = varbinview.bytes_at(i);
            let s = std::str::from_utf8(&str_bytes)
                .map_err(|e| vortex_err!("uuid_from_string: invalid UTF-8 at row {i}: {e}"))?;

            let parsed = uuid::Uuid::parse_str(s)
                .map_err(|e| vortex_err!("uuid_from_string: invalid UUID at row {i}: {e}"))?;

            bytes[i * UUID_BYTE_LEN..(i + 1) * UUID_BYTE_LEN].copy_from_slice(parsed.as_bytes());
        }

        // Build the flat u8 elements array.
        let elements: ArrayRef = Buffer::copy_from(&bytes).into_array();

        // Wrap in FixedSizeList and Extension.
        let fsl = FixedSizeListArray::new(elements, UUID_BYTE_LEN as u32, validity, row_count);
        let ext_dtype = ExtDType::<Uuid>::try_new(UuidMetadata, fsl.dtype().clone())?.erased();

        Ok(ExtensionArray::new(ext_dtype, fsl.into_array()).into_array())
    }

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        // Output validity is the same as the input.
        Ok(Some(expression.child(0).validity()?))
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        // Invalid UUID strings cause errors.
        true
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::DynArray;
    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::arrays::ScalarFnArray;
    use crate::arrays::VarBinViewArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::extension::ExtVTable;
    use crate::extension::uuid::Uuid;
    use crate::extension::uuid::UuidMetadata;
    use crate::extension::uuid::vtable::UUID_BYTE_LEN;
    use crate::scalar_fn::EmptyOptions;
    use crate::scalar_fn::ScalarFn;
    use crate::scalar_fn::fns::uuid_from_string::UuidFromString;

    /// Builds a string array from the given values, with nullable support.
    fn string_array(values: &[Option<&str>]) -> ArrayRef {
        VarBinViewArray::from_iter_nullable_str(values.iter().copied()).into_array()
    }

    /// Evaluates `uuid_from_string` and returns the resulting extension array.
    fn eval_uuid_from_string(input: ArrayRef, len: usize) -> VortexResult<ArrayRef> {
        let scalar_fn = ScalarFn::new(UuidFromString, EmptyOptions).erased();
        let result = ScalarFnArray::try_new(scalar_fn, vec![input], len)?;
        result.to_canonical().map(|c| c.into_array())
    }

    /// Extracts the flat u8 bytes from a UUID extension array.
    fn extract_uuid_bytes(array: &ArrayRef) -> Vec<u8> {
        let ext = array.to_extension();
        let fsl = ext.storage().to_fixed_size_list();
        let prim = fsl.elements().to_primitive();
        prim.as_slice::<u8>().to_vec()
    }

    #[test]
    fn parse_single_uuid() -> VortexResult<()> {
        let input = string_array(&[Some("550e8400-e29b-41d4-a716-446655440000")]);
        let result = eval_uuid_from_string(input, 1)?;

        let expected = uuid::Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000")
            .map_err(|e| vortex_error::vortex_err!("{e}"))?;

        let bytes = extract_uuid_bytes(&result);
        assert_eq!(&bytes, expected.as_bytes());
        Ok(())
    }

    #[test]
    fn parse_multiple_uuids() -> VortexResult<()> {
        let uuids = [
            "550e8400-e29b-41d4-a716-446655440000",
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
            "f47ac10b-58cc-4372-a567-0e02b2c3d479",
        ];
        let input = string_array(&uuids.iter().map(|s| Some(*s)).collect::<Vec<_>>());
        let result = eval_uuid_from_string(input, 3)?;

        let bytes = extract_uuid_bytes(&result);
        for (i, uuid_str) in uuids.iter().enumerate() {
            let expected =
                uuid::Uuid::parse_str(uuid_str).map_err(|e| vortex_error::vortex_err!("{e}"))?;
            assert_eq!(&bytes[i * 16..(i + 1) * 16], expected.as_bytes());
        }
        Ok(())
    }

    #[test]
    fn parse_invalid_uuid_errors() {
        let input = string_array(&[Some("not-a-uuid")]);
        let result = eval_uuid_from_string(input, 1);
        assert!(result.is_err());
    }

    #[test]
    fn parse_null_input_produces_null() -> VortexResult<()> {
        let input = string_array(&[
            Some("550e8400-e29b-41d4-a716-446655440000"),
            None,
            Some("6ba7b810-9dad-11d1-80b4-00c04fd430c8"),
        ]);
        let result = eval_uuid_from_string(input, 3)?;

        // Row 1 should be null.
        assert!(result.is_valid(0)?);
        assert!(result.is_invalid(1)?);
        assert!(result.is_valid(2)?);
        Ok(())
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "UUID_BYTE_LEN always fits both usize and u32"
    )]
    #[test]
    fn storage_array_structure() -> VortexResult<()> {
        // Note that this test assumes that the storage type is a `FixedSizeList`. That will likely
        // change in the future.

        let input = string_array(&[
            Some("550e8400-e29b-41d4-a716-446655440000"),
            None,
            Some("6ba7b810-9dad-11d1-80b4-00c04fd430c8"),
        ]);
        let result = eval_uuid_from_string(input, 3)?;

        // The result should be an extension array.
        let ext = result.to_extension();
        assert_eq!(ext.ext_dtype().id().as_ref(), "vortex.uuid");
        assert_eq!(ext.len(), 3);

        // The storage should be a FixedSizeList of u8 with size 16.
        let fsl = ext.storage().to_fixed_size_list();
        assert_eq!(fsl.len(), 3);
        assert_eq!(fsl.list_size(), UUID_BYTE_LEN as u32);

        // The elements should be a flat u8 primitive array of length 3 * 16 = 48.
        let prim = fsl.elements().to_primitive();
        assert_eq!(prim.len(), 3 * UUID_BYTE_LEN);
        assert_eq!(
            prim.dtype(),
            &DType::Primitive(PType::U8, Nullability::NonNullable)
        );

        // Validity on the FSL should match the input: valid, null, valid.
        assert!(fsl.is_valid(0)?);
        assert!(fsl.is_invalid(1)?);
        assert!(fsl.is_valid(2)?);

        // Verify the byte content of the two valid UUIDs.
        let bytes = prim.as_slice::<u8>();
        let expected_0 = uuid::Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000")
            .map_err(|e| vortex_error::vortex_err!("{e}"))?;
        let expected_2 = uuid::Uuid::parse_str("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
            .map_err(|e| vortex_error::vortex_err!("{e}"))?;
        assert_eq!(&bytes[0..UUID_BYTE_LEN], expected_0.as_bytes());
        assert_eq!(
            &bytes[2 * UUID_BYTE_LEN..3 * UUID_BYTE_LEN],
            expected_2.as_bytes()
        );

        Ok(())
    }

    #[test]
    fn unpack_native_from_parsed() -> VortexResult<()> {
        let input = string_array(&[Some("550e8400-e29b-41d4-a716-446655440000")]);
        let result = eval_uuid_from_string(input, 1)?;

        let scalar = result.scalar_at(0)?;
        let ext_scalar = scalar.as_extension();
        let storage_scalar = ext_scalar.to_storage_scalar();
        let storage_value = storage_scalar
            .value()
            .ok_or_else(|| vortex_error::vortex_err!("expected non-null scalar"))?;

        let native = Uuid.unpack_native(
            &UuidMetadata,
            ext_scalar.ext_dtype().storage_dtype(),
            storage_value,
        )?;
        assert_eq!(native.to_string(), "550e8400-e29b-41d4-a716-446655440000");
        Ok(())
    }
}
