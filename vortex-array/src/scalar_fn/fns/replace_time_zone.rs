// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reinterpret timestamp wall times in a different timezone.

use std::fmt;
use std::sync::Arc;

use jiff::Timestamp as JiffTimestamp;
use jiff::tz::AmbiguousOffset;
use jiff::tz::TimeZone;
use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ExtensionArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::expr::BoundExpression;
use crate::expr::bound;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;
use crate::extension::datetime::TimestampOptions;
use crate::proto::expr as pb;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::literal::Literal;

/// Timezone replacement options. The ambiguity policy is a UTF-8 expression child.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReplaceTimeZoneOptions {
    /// Destination timezone, or `None` to remove the timezone while preserving wall time.
    pub time_zone: Option<Arc<str>>,
    /// Return null for nonexistent wall times instead of raising an error.
    pub null_on_non_existent: bool,
}

impl fmt::Display for ReplaceTimeZoneOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.time_zone {
            Some(tz) => write!(f, "tz={tz}")?,
            None => write!(f, "tz=none")?,
        }
        write!(f, ", null_on_non_existent={}", self.null_on_non_existent)
    }
}

/// Replace a timestamp's timezone, preserving its local wall time.
///
/// The second child selects `raise`, `earliest`, `latest`, or `null` for ambiguous times.
#[derive(Clone)]
pub struct ReplaceTimeZone;

impl ScalarFnVTable for ReplaceTimeZone {
    type Options = ReplaceTimeZoneOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.replace_time_zone");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::ReplaceTimeZoneOpts {
                time_zone: options.time_zone.as_ref().map(ToString::to_string),
                null_on_non_existent: options.null_on_non_existent,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let options = pb::ReplaceTimeZoneOpts::decode(metadata)?;
        Ok(ReplaceTimeZoneOptions {
            time_zone: options.time_zone.map(Arc::from),
            null_on_non_existent: options.null_on_non_existent,
        })
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            1 => ChildName::from("ambiguous"),
            _ => unreachable!("Invalid child index {child_idx} for replace_time_zone()"),
        }
    }

    fn simplify(
        &self,
        options: &Self::Options,
        expr: &BoundExpression,
    ) -> VortexResult<Option<BoundExpression>> {
        let (Some(value), Some(policy)) = (
            expr.child(0).as_opt::<Literal>(),
            expr.child(1).as_opt::<Literal>(),
        ) else {
            return Ok(None);
        };
        // Fold to a literal so stats pruning sees a bare literal operand. A failing replacement
        // (e.g. a raised ambiguity) is left in place so the error surfaces at execution time.
        Ok(replace_scalar(options, value, policy).ok().map(bound::lit))
    }

    fn return_dtype(&self, options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let input = timestamp_options(&arg_dtypes[0])?;
        if input.unit == TimeUnit::Days {
            vortex_bail!("Timestamp cannot use day units");
        }
        if !matches!(arg_dtypes[1], DType::Utf8(_) | DType::Null) {
            vortex_bail!(
                "replace_time_zone() requires a UTF-8 ambiguity policy, got {}",
                arg_dtypes[1]
            );
        }
        resolve_zone(options.time_zone.as_deref())?;
        // A row's ambiguity policy can request null even for non-nullable children.
        Ok(DType::Extension(
            Timestamp::new_with_tz(input.unit, options.time_zone.clone(), Nullability::Nullable)
                .erased(),
        ))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let ambiguous = args.get(1)?;
        let dtype =
            self.return_dtype(options, &[input.dtype().clone(), ambiguous.dtype().clone()])?;
        let metadata = timestamp_options(input.dtype())?.clone();
        let source = resolve_zone(metadata.tz.as_deref())?;
        let target = resolve_zone(options.time_zone.as_deref())?;
        let convert = |value, policy: &str| {
            replace(
                value,
                metadata.unit,
                &source,
                &target,
                policy,
                options.null_on_non_existent,
            )
        };

        if let (Some(value), Some(policy)) = (input.as_constant(), ambiguous.as_constant()) {
            let scalar = replace_scalar(options, &value, &policy)?;
            return Ok(ConstantArray::new(scalar, args.row_count()).into_array());
        }

        if matches!(ambiguous.dtype(), DType::Null) {
            return Ok(ConstantArray::new(Scalar::null(dtype), args.row_count()).into_array());
        }
        let input = input.execute::<ExtensionArray>(ctx)?;
        let storage = input
            .storage_array()
            .clone()
            .execute::<PrimitiveArray>(ctx)?;
        let values = storage.to_buffer::<i64>();
        let valid = storage.validity()?.execute_mask(values.len(), ctx)?;
        let policies = ambiguous.execute::<VarBinViewArray>(ctx)?;
        let policies_valid = policies.validity()?.execute_mask(values.len(), ctx)?;
        let mut output = Vec::with_capacity(values.len());
        for (i, ((value, valid), policy_valid)) in values
            .iter()
            .zip(valid.iter())
            .zip(policies_valid.iter())
            .enumerate()
        {
            output.push(if valid && policy_valid {
                let bytes = policies.bytes_at(i);
                let policy = std::str::from_utf8(&bytes)
                    .map_err(|e| vortex_err!("Invalid ambiguity policy: {e}"))?;
                convert(*value, policy)?
            } else {
                None
            });
        }
        let DType::Extension(ext) = dtype else {
            unreachable!()
        };
        let storage = PrimitiveArray::from_option_iter(output)
            .into_array()
            .cast(DType::Primitive(PType::I64, Nullability::Nullable))?;
        Ok(ExtensionArray::new(ext, storage).into_array())
    }
}

/// Replace the timezone of a single timestamp scalar using a scalar ambiguity policy.
fn replace_scalar(
    options: &ReplaceTimeZoneOptions,
    value: &Scalar,
    policy: &Scalar,
) -> VortexResult<Scalar> {
    let DType::Extension(ext) =
        ReplaceTimeZone.return_dtype(options, &[value.dtype().clone(), policy.dtype().clone()])?
    else {
        unreachable!("replace_time_zone() returns a timestamp")
    };
    let metadata = timestamp_options(value.dtype())?;
    let result = if value.is_null() || policy.is_null() {
        None
    } else {
        let storage = value.as_extension().to_storage_scalar();
        let policy = policy
            .as_utf8()
            .value()
            .ok_or_else(|| vortex_err!("Missing ambiguity policy"))?;
        replace(
            i64::try_from(&storage)?,
            metadata.unit,
            &resolve_zone(metadata.tz.as_deref())?,
            &resolve_zone(options.time_zone.as_deref())?,
            policy.as_str(),
            options.null_on_non_existent,
        )?
    };
    let storage = result.map_or_else(
        || Scalar::null(DType::Primitive(PType::I64, Nullability::Nullable)),
        |v| Scalar::primitive(v, Nullability::Nullable),
    );
    Ok(Scalar::extension_ref(ext, storage))
}

fn timestamp_options(dtype: &DType) -> VortexResult<&TimestampOptions> {
    if let DType::Extension(ext) = dtype
        && let Some(options) = ext.metadata_opt::<Timestamp>()
    {
        return Ok(options);
    }
    vortex_bail!("replace_time_zone() requires Timestamp, got {dtype}")
}

fn resolve_zone(name: Option<&str>) -> VortexResult<TimeZone> {
    Ok(match name {
        Some(name) => TimeZone::get(name)?,
        None => TimeZone::UTC,
    })
}

fn replace(
    value: i64,
    unit: TimeUnit,
    source: &TimeZone,
    target: &TimeZone,
    policy: &str,
    null_on_non_existent: bool,
) -> VortexResult<Option<i64>> {
    if !matches!(policy, "raise" | "earliest" | "latest" | "null") {
        vortex_bail!(
            "Invalid ambiguity policy {policy:?}: expected raise, earliest, latest, or null"
        );
    }
    // Keeping the same timezone does not introduce a new ambiguity to reject.
    if source == target && policy == "raise" {
        return Ok(Some(value));
    }
    let timestamp = match unit {
        TimeUnit::Seconds => JiffTimestamp::from_second(value)?,
        TimeUnit::Milliseconds => JiffTimestamp::from_millisecond(value)?,
        TimeUnit::Microseconds => JiffTimestamp::from_microsecond(value)?,
        TimeUnit::Nanoseconds => JiffTimestamp::from_nanosecond(i128::from(value))?,
        TimeUnit::Days => vortex_bail!("Timestamp cannot use day units"),
    };
    let wall_time = source.to_datetime(timestamp);
    let ambiguous = target.to_ambiguous_timestamp(wall_time);
    let timestamp = match ambiguous.offset() {
        AmbiguousOffset::Unambiguous { .. } => ambiguous.unambiguous()?,
        AmbiguousOffset::Gap { .. } if null_on_non_existent => return Ok(None),
        AmbiguousOffset::Gap { .. } => ambiguous.unambiguous()?,
        AmbiguousOffset::Fold { .. } => match policy {
            "earliest" => ambiguous.earlier()?,
            "latest" => ambiguous.later()?,
            "null" => return Ok(None),
            _ => ambiguous.unambiguous()?,
        },
    };
    Ok(Some(match unit {
        TimeUnit::Seconds => timestamp.as_second(),
        TimeUnit::Milliseconds => timestamp.as_millisecond(),
        TimeUnit::Microseconds => timestamp.as_microsecond(),
        TimeUnit::Nanoseconds => i64::try_from(timestamp.as_nanosecond())
            .map_err(|e| vortex_err!("Timestamp overflow: {e}"))?,
        TimeUnit::Days => unreachable!(),
    }))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use super::ReplaceTimeZone;
    use super::ReplaceTimeZoneOptions;
    use super::replace;
    use super::resolve_zone;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::StructFields;
    use crate::expr::lit;
    use crate::expr::replace_time_zone;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::scalar::Scalar;
    use crate::scalar_fn::ScalarFnVTable;
    use crate::scalar_fn::fns::literal::Literal;

    #[rstest]
    #[case(TimeUnit::Seconds, 1)]
    #[case(TimeUnit::Milliseconds, 1_000)]
    #[case(TimeUnit::Microseconds, 1_000_000)]
    #[case(TimeUnit::Nanoseconds, 1_000_000_000)]
    fn preserves_units_and_fractional_values(
        #[case] unit: TimeUnit,
        #[case] scale: i64,
    ) -> VortexResult<()> {
        let utc = resolve_zone(None)?;
        let new_york = resolve_zone(Some("America/New_York"))?;
        let value = 1_705_320_000 * scale + scale / 2;
        assert_eq!(
            replace(value, unit, &utc, &new_york, "raise", false)?,
            Some(value + 18_000 * scale)
        );
        assert_eq!(
            replace(
                value + 18_000 * scale,
                unit,
                &new_york,
                &utc,
                "raise",
                false
            )?,
            Some(value)
        );
        Ok(())
    }

    #[test]
    fn resolves_folds_and_gaps_separately() -> VortexResult<()> {
        let utc = resolve_zone(None)?;
        let new_york = resolve_zone(Some("America/New_York"))?;
        let fold = "2024-11-03T01:30:00Z"
            .parse::<jiff::Timestamp>()?
            .as_second();
        let gap = "2024-03-10T02:30:00Z"
            .parse::<jiff::Timestamp>()?
            .as_second();
        let convert = |value, policy, null_on_non_existent| {
            replace(
                value,
                TimeUnit::Seconds,
                &utc,
                &new_york,
                policy,
                null_on_non_existent,
            )
        };
        assert_eq!(convert(fold, "earliest", false)?, Some(fold + 14_400));
        assert_eq!(convert(fold, "latest", false)?, Some(fold + 18_000));
        assert_eq!(convert(fold, "null", false)?, None);
        assert!(convert(fold, "raise", false).is_err());
        assert_eq!(convert(gap, "raise", true)?, None);
        // Ambiguity policies must not resolve nonexistent wall times.
        assert!(convert(gap, "earliest", false).is_err());
        Ok(())
    }

    #[test]
    fn rejects_invalid_policy_and_timestamp_overflow() -> VortexResult<()> {
        let utc = resolve_zone(None)?;
        assert!(replace(0, TimeUnit::Seconds, &utc, &utc, "invalid", false).is_err());
        let new_york = resolve_zone(Some("America/New_York"))?;
        assert!(
            replace(
                i64::MAX,
                TimeUnit::Nanoseconds,
                &utc,
                &new_york,
                "raise",
                false
            )
            .is_err()
        );
        assert!(resolve_zone(Some("Not/A/Timezone")).is_err());
        Ok(())
    }

    #[rstest]
    #[case(None, false)]
    #[case(Some("America/New_York"), true)]
    fn options_round_trip(
        #[case] time_zone: Option<&str>,
        #[case] null_on_non_existent: bool,
    ) -> VortexResult<()> {
        let options = ReplaceTimeZoneOptions {
            time_zone: time_zone.map(Into::into),
            null_on_non_existent,
        };
        let metadata = ReplaceTimeZone.serialize(&options)?.unwrap();
        assert_eq!(
            ReplaceTimeZone.deserialize(&metadata, &VortexSession::empty())?,
            options
        );
        Ok(())
    }

    #[test]
    fn simplify_folds_literal_input() -> VortexResult<()> {
        let timestamp = |tz: Option<&str>| {
            Timestamp::new_with_tz(
                TimeUnit::Microseconds,
                tz.map(Into::into),
                Nullability::Nullable,
            )
            .erased()
        };
        let micros = Scalar::primitive(1_704_153_600_000_000i64, Nullability::Nullable);
        let expr = replace_time_zone(
            lit(Scalar::extension_ref(timestamp(None), micros.clone())),
            lit("earliest"),
            ReplaceTimeZoneOptions {
                time_zone: Some("UTC".into()),
                null_on_non_existent: false,
            },
        );
        let scope = DType::Struct(StructFields::empty(), Nullability::NonNullable);
        let optimized = expr.bind(&scope)?.optimize()?;

        // Stats pruning only matches bare literal operands, so the replacement must fold.
        let scalar = optimized
            .as_opt::<Literal>()
            .ok_or_else(|| vortex_err!("expected a bare literal, got {optimized}"))?;
        assert_eq!(
            scalar,
            &Scalar::extension_ref(timestamp(Some("UTC")), micros)
        );
        Ok(())
    }
}
