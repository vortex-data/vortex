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
use vortex_array::expr::Expression;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::Uuid;
use crate::UuidMetadata;
use crate::vtable::UUID_BYTE_LEN;
use crate::vtable::check_version;

/// Parse UTF-8 text as a UUID and enforce the optional version in `metadata`.
pub fn parse_uuid(child: Expression, metadata: UuidMetadata) -> Expression {
    ParseUuid.new_expr(metadata, [child])
}

/// Format UUID values as lowercase hyphenated text.
///
/// A stored value that violates the dtype's version constraint is an error.
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
        let storage = FixedSizeListArray::try_new(elements.into_array(), 16, validity, len)?;
        let target = dtype.as_extension_opt().vortex_expect("UUID target dtype");
        Ok(ExtensionArray::try_new(target.clone(), storage.into_array())?.into_array())
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
        vortex_ensure!(bytes.is_empty(), "Expected empty UUID formatting options");
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
            .vortex_expect("validated UUID input")
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
        let mut text = [0; 36];
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
}

#[cfg(test)]
mod tests {
    use uuid::Version;
    use vortex_array::VortexSessionExecute;
    use vortex_array::expr::proto::ExprSerializeProtoExt;
    use vortex_array::expr::root;
    use vortex_array::test_harness::ExpectedOutcome;
    use vortex_array::test_harness::assert_execution_paths;

    use super::*;

    #[test]
    fn test_uuid_text_conversion() -> VortexResult<()> {
        let session = VortexSession::empty();
        crate::initialize(&session);
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
        let expected = VarBinViewArray::from_iter_nullable_str([
            Some("550e8400-e29b-41d4-a716-446655440000"),
            None,
        ])
        .into_array();
        assert_execution_paths(
            &[input],
            |inputs| inputs[0].clone().apply(&operation),
            ExpectedOutcome::Array(expected),
            &mut ctx,
        );
        let invalid =
            VarBinViewArray::from_iter_str(["malformed", "00000000-0000-0000-0000-000000000000"])
                .into_array();
        assert_execution_paths(
            &[invalid],
            |inputs| inputs[0].clone().apply(&operation),
            ExpectedOutcome::Error,
            &mut ctx,
        );
        Ok(())
    }
}
