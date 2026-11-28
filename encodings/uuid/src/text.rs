// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Typed parsing and formatting for UUIDs.
//!
//! Both directions check the version declared by the UUID dtype when one is present.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_expr::Expression;
use vortex_expr::ScalarFnExprExt;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::Uuid;
use crate::UuidMetadata;
use crate::vtable::UUID_BYTE_LEN;
use crate::vtable::check_version;

/// Parse UTF-8 text as a [`Uuid`] extension array and enforce the optional version in `metadata`.
///
/// The input **must** have a [`DType::Utf8`] dtype, or binding fails. Accepted text formats are
/// hyphenated, simple, braced, and URN, parsed case-insensitively. Nulls stay null, and the output
/// preserves the input's nullability. At execution, malformed UUID text or a parsed value whose
/// version disagrees with `metadata.version` returns an error.
pub fn parse_uuid(child: Expression, metadata: UuidMetadata) -> Expression {
    ParseUuid.new_expr(metadata, [child])
}

/// Format [`Uuid`] values as lowercase hyphenated UTF-8 text.
///
/// The input **must** have a [`Uuid`] extension dtype, or binding fails. Nulls stay null, and the
/// output preserves the input's nullability. At execution, a stored non-null value whose version
/// disagrees with the dtype's declared version returns an error.
pub fn format_uuid(child: Expression) -> Expression {
    FormatUuid.new_expr(EmptyOptions, [child])
}

/// Bind UTF-8 input and the target UUID version constraint.
#[derive(Clone)]
pub(crate) struct ParseUuid;

impl ScalarFnVTable for ParseUuid {
    type Options = UuidMetadata;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.parse_uuid");
        *ID
    }

    fn arity(&self, _: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _: &Self::Options, _: usize) -> ChildName {
        ChildName::from("input")
    }

    fn serialize(&self, metadata: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(Uuid.serialize_metadata(metadata)?))
    }

    fn deserialize(&self, bytes: &[u8], _: &VortexSession) -> VortexResult<Self::Options> {
        Uuid.deserialize_metadata(bytes)
    }

    fn return_dtype(&self, metadata: &Self::Options, inputs: &[DType]) -> VortexResult<DType> {
        let input = &inputs[0];
        vortex_ensure!(
            matches!(input, DType::Utf8(_)),
            "Expected UTF-8 text, got {input}"
        );

        Ok(DType::Extension(
            ExtDType::try_with_vtable(Uuid, metadata.clone(), input.nullability())?.erased(),
        ))
    }

    fn execute(
        &self,
        metadata: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let dtype = self.return_dtype(metadata, std::slice::from_ref(input.dtype()))?;
        if input.is_empty() {
            return Ok(Canonical::empty(&dtype).into_array());
        }

        let len = input.len();
        let input = input.execute::<VarBinViewArray>(ctx)?;
        let validity = input.validity()?;
        let mask = validity.execute_mask(len, ctx)?;
        let mut bytes = BufferMut::<u8>::with_capacity(
            len.checked_mul(UUID_BYTE_LEN)
                .ok_or_else(|| vortex_err!("UUID storage length overflows for {len} rows"))?,
        );

        for (index, valid) in mask.iter().enumerate() {
            let value = if valid {
                let uuid = uuid::Uuid::try_parse_ascii(&input.bytes_at(index))
                    .map_err(|e| vortex_err!("Invalid UUID at row {index}: {e}"))?;
                check_version(uuid, metadata)?;
                uuid.into_bytes()
            } else {
                [0; UUID_BYTE_LEN]
            };
            bytes.extend_from_slice(&value);
        }

        let elements = PrimitiveArray::new(bytes.freeze(), Validity::NonNullable);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "UUID_BYTE_LEN always fits u32"
        )]
        let storage = FixedSizeListArray::try_new(
            elements.into_array(),
            UUID_BYTE_LEN as u32,
            validity,
            len,
        )?;
        let target = dtype
            .as_extension_opt()
            .vortex_expect("`return_dtype` above returns a `Uuid` extension dtype");

        Ok(ExtensionArray::try_new(target.clone(), storage.into_array())?.into_array())
    }

    fn is_strict(&self, _: &Self::Options) -> bool {
        true
    }
}

/// Validate and format each visible value of a UUID array.
#[derive(Clone)]
pub(crate) struct FormatUuid;

impl ScalarFnVTable for FormatUuid {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.format_uuid");
        *ID
    }

    fn arity(&self, _: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _: &Self::Options, _: usize) -> ChildName {
        ChildName::from("input")
    }

    fn serialize(&self, _: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(&self, bytes: &[u8], _: &VortexSession) -> VortexResult<Self::Options> {
        vortex_ensure_eq!(bytes.len(), 0, "Expected empty UUID formatting options");

        Ok(EmptyOptions)
    }

    fn return_dtype(&self, _: &Self::Options, inputs: &[DType]) -> VortexResult<DType> {
        let input = &inputs[0];
        vortex_ensure!(
            input.as_extension_opt().is_some_and(|ext| ext.is::<Uuid>()),
            "Expected UUID, got {input}"
        );

        Ok(DType::Utf8(input.nullability()))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let dtype = self.return_dtype(options, std::slice::from_ref(input.dtype()))?;
        if input.is_empty() {
            return Ok(Canonical::empty(&dtype).into_array());
        }

        let len = input.len();
        let metadata = input
            .dtype()
            .as_extension_opt()
            .and_then(|ext| ext.metadata_opt::<Uuid>())
            .vortex_expect("`return_dtype` above accepts only `Uuid` extensions")
            .clone();
        let input = input.execute::<ExtensionArray>(ctx)?;
        let storage = input
            .storage_array()
            .clone()
            .execute::<FixedSizeListArray>(ctx)?;
        let values = storage
            .elements()
            .clone()
            .execute::<PrimitiveArray>(ctx)?
            .into_buffer::<u8>();
        let mask = storage.validity()?.execute_mask(len, ctx)?;
        let mut builder = VarBinViewBuilder::with_capacity_in(dtype, len, ctx.allocator().clone());
        let mut text = [0; uuid::fmt::Hyphenated::LENGTH];
        let (uuids, remainder) = values.as_chunks::<UUID_BYTE_LEN>();
        debug_assert!(remainder.is_empty());
        debug_assert_eq!(uuids.len(), len);

        for (&bytes, valid) in uuids.iter().zip(mask.iter()) {
            if !valid {
                builder.append_null();
                continue;
            }
            let uuid = uuid::Uuid::from_bytes(bytes);
            check_version(uuid, &metadata)?;
            builder.append_value(uuid.hyphenated().encode_lower(&mut text));
        }

        Ok(builder.finish_into_varbinview().into_array())
    }

    fn is_strict(&self, _: &Self::Options) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use uuid::Version;
    use vortex_array::VortexSessionExecute;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::test_harness::ExpectedOutcome;
    use vortex_array::test_harness::assert_execution_paths;
    use vortex_buffer::Buffer;
    use vortex_expr::ArrayExprExt;
    use vortex_expr::proto::ExprSerializeProtoExt;
    use vortex_expr::root;

    use super::*;

    const V4_UUID_STR: &str = "550e8400-e29b-41d4-a716-446655440000";
    const V4_UUID_BYTES: [u8; UUID_BYTE_LEN] = [
        0x55, 0x0e, 0x84, 0x00, 0xe2, 0x9b, 0x41, 0xd4, 0xa7, 0x16, 0x44, 0x66, 0x55, 0x44, 0x00,
        0x00,
    ];
    const NIL_UUID_STR: &str = "00000000-0000-0000-0000-000000000000";
    const NIL_UUID_BYTES: [u8; UUID_BYTE_LEN] = [0; UUID_BYTE_LEN];

    fn uuid_session() -> VortexSession {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "UUID_BYTE_LEN always fits u32"
    )]
    fn uuid_array(
        rows: &[Option<[u8; UUID_BYTE_LEN]>],
        metadata: UuidMetadata,
        nullability: Nullability,
    ) -> VortexResult<ArrayRef> {
        let flat_bytes = rows
            .iter()
            .flat_map(|row| row.unwrap_or([0; UUID_BYTE_LEN]))
            .collect::<Buffer<u8>>();
        let validity = match nullability {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => rows.iter().map(Option::is_some).collect(),
        };
        let storage = FixedSizeListArray::try_new(
            PrimitiveArray::new(flat_bytes, Validity::NonNullable).into_array(),
            UUID_BYTE_LEN as u32,
            validity,
            rows.len(),
        )?;
        let ext_dtype = ExtDType::try_with_vtable(Uuid, metadata, nullability)?;

        Ok(ExtensionArray::try_new(ext_dtype.erased(), storage.into_array())?.into_array())
    }

    #[test]
    fn test_uuid_text_conversion() -> VortexResult<()> {
        let session = uuid_session();
        let mut ctx = session.create_execution_ctx();
        let operation = format_uuid(parse_uuid(
            root(),
            UuidMetadata {
                version: Some(Version::Random),
            },
        ));
        let operation = Expression::from_proto(&operation.serialize_proto()?, &session)?;

        let input = VarBinViewArray::from_iter_nullable_str([
            Some("550E8400-E29B-41D4-A716-446655440000"),
            None,
        ])
        .into_array();
        let expected =
            VarBinViewArray::from_iter_nullable_str([Some(V4_UUID_STR), None]).into_array();

        assert_execution_paths(
            &[input],
            |inputs| inputs[0].clone().apply(&operation),
            ExpectedOutcome::Array(expected),
            &mut ctx,
        );

        let invalid = VarBinViewArray::from_iter_str(["malformed", NIL_UUID_STR]).into_array();

        assert_execution_paths(
            &[invalid],
            |inputs| inputs[0].clone().apply(&operation),
            ExpectedOutcome::Error,
            &mut ctx,
        );

        Ok(())
    }

    #[test]
    fn test_parse_uuid_accepted_formats_non_nullable() -> VortexResult<()> {
        let session = uuid_session();
        let mut ctx = session.create_execution_ctx();
        let metadata = UuidMetadata {
            version: Some(Version::Random),
        };
        let operation = parse_uuid(root(), metadata.clone());
        let input = VarBinViewArray::from_iter_str([
            "550E8400-E29B-41D4-A716-446655440000", // hyphenated uppercase
            "550e8400e29b41d4a716446655440000",     // simple
            "{550e8400-e29b-41d4-a716-446655440000}", // braced
            "urn:uuid:550e8400-e29b-41d4-a716-446655440000", // URN
        ])
        .into_array();
        let expected = uuid_array(
            &[
                Some(V4_UUID_BYTES),
                Some(V4_UUID_BYTES),
                Some(V4_UUID_BYTES),
                Some(V4_UUID_BYTES),
            ],
            metadata,
            Nullability::NonNullable,
        )?;

        assert_execution_paths(
            &[input],
            |inputs| inputs[0].clone().apply(&operation),
            ExpectedOutcome::Array(expected),
            &mut ctx,
        );

        Ok(())
    }

    #[test]
    fn test_parse_uuid_without_version_accepts_mixed_versions() -> VortexResult<()> {
        let session = uuid_session();
        let mut ctx = session.create_execution_ctx();
        let metadata = UuidMetadata { version: None };
        let operation = parse_uuid(root(), metadata.clone());
        let input =
            VarBinViewArray::from_iter_nullable_str([Some(V4_UUID_STR), None, Some(NIL_UUID_STR)])
                .into_array();
        let expected = uuid_array(
            &[Some(V4_UUID_BYTES), None, Some(NIL_UUID_BYTES)],
            metadata,
            Nullability::Nullable,
        )?;

        assert_execution_paths(
            &[input],
            |inputs| inputs[0].clone().apply(&operation),
            ExpectedOutcome::Array(expected),
            &mut ctx,
        );

        Ok(())
    }

    #[test]
    fn test_parse_uuid_empty_and_all_null() -> VortexResult<()> {
        let session = uuid_session();
        let mut ctx = session.create_execution_ctx();
        let metadata = UuidMetadata {
            version: Some(Version::Random),
        };
        let operation = parse_uuid(root(), metadata.clone());

        let empty_input = VarBinViewArray::from_iter_str(std::iter::empty::<&str>()).into_array();
        let empty_expected = uuid_array(&[], metadata.clone(), Nullability::NonNullable)?;

        assert_execution_paths(
            &[empty_input],
            |inputs| inputs[0].clone().apply(&operation),
            ExpectedOutcome::Array(empty_expected),
            &mut ctx,
        );

        let all_null_input =
            VarBinViewArray::from_iter_nullable_str([None::<&str>, None]).into_array();
        let all_null_expected = uuid_array(&[None, None], metadata, Nullability::Nullable)?;

        assert_execution_paths(
            &[all_null_input],
            |inputs| inputs[0].clone().apply(&operation),
            ExpectedOutcome::Array(all_null_expected),
            &mut ctx,
        );

        Ok(())
    }

    #[rstest]
    #[case::malformed("not-a-uuid", "Invalid UUID at row 0")]
    #[case::version_mismatch(NIL_UUID_STR, "UUID version mismatch")]
    fn test_parse_uuid_errors(
        #[case] value: &str,
        #[case] expected_message: &str,
    ) -> VortexResult<()> {
        let session = uuid_session();
        let mut ctx = session.create_execution_ctx();
        let operation = parse_uuid(
            root(),
            UuidMetadata {
                version: Some(Version::Random),
            },
        );
        let input = VarBinViewArray::from_iter_str([value]).into_array();

        let err = input
            .clone()
            .apply(&operation)?
            .execute::<ExtensionArray>(&mut ctx)
            .expect_err("expected parse_uuid to fail");
        assert!(
            err.to_string().contains(expected_message),
            "expected error containing {expected_message:?}, got {err}"
        );

        assert_execution_paths(
            &[input],
            |inputs| inputs[0].clone().apply(&operation),
            ExpectedOutcome::Error,
            &mut ctx,
        );

        Ok(())
    }

    #[test]
    fn test_format_uuid_rejects_stored_version_mismatch() -> VortexResult<()> {
        let session = uuid_session();
        let mut ctx = session.create_execution_ctx();
        let operation = format_uuid(root());
        let invalid = uuid_array(
            &[Some(NIL_UUID_BYTES)],
            UuidMetadata {
                version: Some(Version::Random),
            },
            Nullability::NonNullable,
        )?;

        let err = invalid
            .apply(&operation)?
            .execute::<VarBinViewArray>(&mut ctx)
            .expect_err("expected format_uuid to reject stored version mismatch");
        assert!(
            err.to_string().contains("UUID version mismatch"),
            "expected version mismatch error, got {err}"
        );

        let with_null = uuid_array(
            &[Some(V4_UUID_BYTES), None],
            UuidMetadata {
                version: Some(Version::Random),
            },
            Nullability::Nullable,
        )?;
        let expected =
            VarBinViewArray::from_iter_nullable_str([Some(V4_UUID_STR), None]).into_array();

        assert_execution_paths(
            &[with_null],
            |inputs| inputs[0].clone().apply(&operation),
            ExpectedOutcome::Array(expected),
            &mut ctx,
        );

        Ok(())
    }

    #[rstest]
    #[case::binary(DType::Binary(Nullability::NonNullable))]
    #[case::primitive(DType::Primitive(PType::U64, Nullability::Nullable))]
    fn test_parse_uuid_return_dtype_rejects_non_utf8(#[case] input: DType) {
        let err = ParseUuid
            .return_dtype(&UuidMetadata::default(), &[input])
            .expect_err("expected parse_uuid to reject non-UTF-8 input");
        assert!(
            err.to_string().contains("Expected UTF-8 text"),
            "unexpected error: {err}"
        );
    }

    #[rstest]
    #[case::utf8(DType::Utf8(Nullability::NonNullable))]
    #[case::storage(Uuid::storage_dtype(&UuidMetadata::default()))]
    fn test_format_uuid_return_dtype_rejects_non_uuid(#[case] input: DType) {
        let err = FormatUuid
            .return_dtype(&EmptyOptions, &[input])
            .expect_err("expected format_uuid to reject non-UUID input");
        assert!(
            err.to_string().contains("Expected UUID"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_format_uuid_deserialize_rejects_non_empty_metadata() {
        let session = uuid_session();
        let err = FormatUuid
            .deserialize(&[1, 2], &session)
            .expect_err("expected FormatUuid::deserialize to reject non-empty bytes");
        let message = err.to_string();
        assert!(
            message.contains("Expected empty UUID formatting options") && message.contains('2'),
            "unexpected error: {message}"
        );
    }
}
