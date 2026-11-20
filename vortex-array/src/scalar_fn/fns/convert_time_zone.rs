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
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ExtensionArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::dtype::DType;
use crate::extension::datetime::Timestamp;
use crate::extension::datetime::TimestampOptions;
use crate::proto::expr as pb;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

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
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::Dict;
    use crate::arrays::DictArray;
    use crate::arrays::ExtensionArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::Nullability;
    use crate::dtype::extension::ExtDTypeRef;
    use crate::expr::Expression;
    use crate::expr::convert_time_zone;
    use crate::expr::proto::ExprSerializeProtoExt;
    use crate::expr::root;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::test_harness::ExpectedOutcome;
    use crate::test_harness::assert_execution_paths;

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
