// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Change the time zone of a timestamp and keep its instant.
//!
//! A timestamp with a time zone stores an instant, so the stored count stays the same and only the
//! zone in the dtype changes. [`replace_time_zone`](crate::expr::replace_time_zone) keeps the
//! local clock time instead, which changes the stored count.

use std::fmt;
use std::sync::Arc;

use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::dtype::DType;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::extension::datetime::TimestampOptions;
use vortex_array::proto::expr as pb;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

/// Options for [`ConvertTimeZone`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConvertTimeZoneOptions {
    /// The target time zone, as an IANA name or a fixed UTC offset.
    pub time_zone: Arc<str>,
}

impl fmt::Display for ConvertTimeZoneOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tz={}", self.time_zone)
    }
}

/// Change the time zone of a [`Timestamp`] and keep its instant.
///
/// See [`convert_time_zone`](crate::expr::convert_time_zone) for the contract.
#[derive(Clone)]
pub struct ConvertTimeZone;

impl ScalarFnVTable for ConvertTimeZone {
    type Options = ConvertTimeZoneOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.convert_time_zone");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::ConvertTimeZoneOpts {
                time_zone: options.time_zone.to_string(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let options = pb::ConvertTimeZoneOpts::decode(metadata)?;
        Ok(ConvertTimeZoneOptions {
            time_zone: Arc::from(options.time_zone),
        })
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, _child_idx: usize) -> ChildName {
        ChildName::from("input")
    }

    fn return_dtype(&self, options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let input = &arg_dtypes[0];
        let Some(TimestampOptions { unit, tz: Some(_) }) = input
            .as_extension_opt()
            .and_then(|ext| ext.metadata_opt::<Timestamp>())
        else {
            vortex_bail!("Expected a timestamp with a time zone, got {input}");
        };

        let target = Timestamp::try_new_with_tz(
            *unit,
            Some(Arc::clone(&options.time_zone)),
            input.nullability(),
        )?;
        Ok(DType::Extension(target.erased()))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let DType::Extension(target) =
            self.return_dtype(options, std::slice::from_ref(input.dtype()))?
        else {
            vortex_panic!("`return_dtype` returns a timestamp dtype");
        };

        // A constant keeps its one stored instant, so the result stays constant.
        if let Some(scalar) = input.as_constant() {
            let output =
                Scalar::try_extension_ref(target, scalar.as_extension().to_storage_scalar())?;
            return Ok(ConstantArray::new(output, input.len()).into_array());
        }

        let input = input.execute::<ExtensionArray>(ctx)?;
        Ok(ExtensionArray::try_new(target, input.storage_array().clone())?.into_array())
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        // Changing the zone keeps every stored instant, so no value can fail.
        true
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Dict;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::extension::ExtDTypeRef;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::extension::datetime::Timestamp;
    use vortex_array::test_harness::ExpectedOutcome;
    use vortex_array::test_harness::assert_execution_paths;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_expr::ArrayExprExt;
    use vortex_expr::Expression;
    use vortex_expr::proto::ExprSerializeProtoExt;
    use vortex_expr::root;
    use vortex_session::VortexSession;

    use crate::expr::convert_time_zone;

    fn zoned(zone: &str, nullability: Nullability) -> ExtDTypeRef {
        Timestamp::new_with_tz(TimeUnit::Seconds, Some(zone.into()), nullability).erased()
    }

    #[test]
    fn test_convert_time_zone_keeps_instants() -> VortexResult<()> {
        let input = ExtensionArray::try_new(
            zoned("America/New_York", Nullability::Nullable),
            PrimitiveArray::from_option_iter([Some(0i64), None, Some(1_710_054_000)]).into_array(),
        )?;
        let expected = ExtensionArray::try_new(
            zoned("UTC", Nullability::Nullable),
            PrimitiveArray::from_option_iter([Some(0i64), None, Some(1_710_054_000)]).into_array(),
        )?;
        let session = VortexSession::empty();
        crate::initialize(&session);
        let operation = convert_time_zone(root(), "UTC");
        let restored = Expression::from_proto(&operation.serialize_proto()?, &session)?;

        assert_execution_paths(
            &[input.into_array()],
            |inputs| inputs[0].clone().apply(&restored),
            ExpectedOutcome::Array(expected.into_array()),
            &mut session.create_execution_ctx(),
        );
        Ok(())
    }

    #[test]
    fn test_convert_time_zone_pushes_into_dictionary_values() -> VortexResult<()> {
        // The null code needs a strict function, and the unused value `1` needs an infallible one.
        let input = DictArray::try_new(
            PrimitiveArray::from_option_iter([Some(0u8), None, Some(0)]).into_array(),
            ExtensionArray::try_new(
                zoned("America/New_York", Nullability::NonNullable),
                buffer![0i64, 1].into_array(),
            )?
            .into_array(),
        )?;
        let mut ctx = VortexSession::empty().create_execution_ctx();

        let result = input
            .into_array()
            .apply(&convert_time_zone(root(), "UTC"))?
            .execute_until::<Dict>(&mut ctx)?;

        assert!(result.is::<Dict>());
        let expected = ExtensionArray::try_new(
            zoned("UTC", Nullability::Nullable),
            PrimitiveArray::from_option_iter([Some(0i64), None, Some(0)]).into_array(),
        )?;
        assert_arrays_eq!(result, expected.into_array(), &mut ctx);
        Ok(())
    }
}
