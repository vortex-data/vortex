// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversion of the per-column statistics stored in a Vortex file footer into
//! `dev.vortex.api.VortexColumnStatistics` objects.
//!
//! A read resolves each statistic through the file's [`DataSource`]. It uses the stored
//! aggregate partials first, and falls back to the finalized legacy statistics only for a
//! statistic that has no partial.

use jni::Env;
use jni::objects::JObject;
use jni::objects::JObjectArray;
use jni::objects::JValue;
use jni::sys::JNI_FALSE;
use jni::sys::JNI_TRUE;
use jni::sys::jlong;
use vortex::aggregate_fn::AggregateFnRef;
use vortex::array::scalar::PValue;
use vortex::array::scalar::Scalar;
use vortex::array::scalar::ScalarValue;
use vortex::array::stats::StatsSet;
use vortex::dtype::DType;
use vortex::dtype::FieldPath;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::stats::Precision;
use vortex::expr::stats::Stat;
use vortex::file::Footer;
use vortex::layout::layouts::file_stats::AggregateStats;
use vortex::scan::DataSource;

use crate::errors::JNIError;

pub(crate) fn checked_jlong(value: u64, name: &str) -> VortexResult<jlong> {
    jlong::try_from(value).map_err(|_| vortex_err!("{name} exceeds Java long range: {value}"))
}

/// Returns an exact count as a `jlong`, or `-1` when the count is not exact.
fn exact_count_jlong(count: Precision<Scalar>, name: &str) -> VortexResult<jlong> {
    count
        .as_exact()
        .and_then(|count| u64::try_from(&count).ok())
        .map(|value| checked_jlong(value, name))
        .transpose()
        .map(|value| value.unwrap_or(-1))
}

fn big_integer<'local>(
    env: &mut Env<'local>,
    value: impl ToString,
) -> Result<JObject<'local>, JNIError> {
    let string = env.new_string(value.to_string())?;
    Ok(env.new_object(
        jni::jni_str!("java/math/BigInteger"),
        jni::jni_sig!("(Ljava/lang/String;)V"),
        &[JValue::Object(string.as_ref())],
    )?)
}

fn scalar_to_java<'local>(
    env: &mut Env<'local>,
    scalar: Scalar,
) -> Result<JObject<'local>, JNIError> {
    if scalar.is_null() {
        return Ok(JObject::null());
    }
    if scalar.dtype().is_extension() {
        return scalar_to_java(env, scalar.as_extension().to_storage_scalar());
    }

    let Some(value) = scalar.value() else {
        return Ok(JObject::null());
    };
    match value {
        ScalarValue::Bool(value) => Ok(env.new_object(
            jni::jni_str!("java/lang/Boolean"),
            jni::jni_sig!("(Z)V"),
            &[JValue::Bool(if *value { JNI_TRUE } else { JNI_FALSE })],
        )?),
        ScalarValue::Primitive(value) => match value {
            PValue::U8(_) | PValue::U16(_) | PValue::I8(_) | PValue::I16(_) | PValue::I32(_) => {
                Ok(env.new_object(
                    jni::jni_str!("java/lang/Integer"),
                    jni::jni_sig!("(I)V"),
                    &[JValue::Int(value.cast::<i32>()?)],
                )?)
            }
            PValue::U32(_) | PValue::I64(_) => Ok(env.new_object(
                jni::jni_str!("java/lang/Long"),
                jni::jni_sig!("(J)V"),
                &[JValue::Long(value.cast::<i64>()?)],
            )?),
            PValue::U64(value) => big_integer(env, value),
            PValue::F16(_) | PValue::F32(_) => Ok(env.new_object(
                jni::jni_str!("java/lang/Float"),
                jni::jni_sig!("(F)V"),
                &[JValue::Float(value.cast::<f32>()?)],
            )?),
            PValue::F64(value) => Ok(env.new_object(
                jni::jni_str!("java/lang/Double"),
                jni::jni_sig!("(D)V"),
                &[JValue::Double(*value)],
            )?),
        },
        ScalarValue::Decimal(value) => {
            let DType::Decimal(decimal_dtype, _) = scalar.dtype() else {
                return Err(JNIError::Vortex(vortex_err!(
                    "decimal statistic has non-decimal dtype {}",
                    scalar.dtype()
                )));
            };
            let unscaled = big_integer(env, value.as_i256())?;
            Ok(env.new_object(
                jni::jni_str!("java/math/BigDecimal"),
                jni::jni_sig!("(Ljava/math/BigInteger;I)V"),
                &[
                    JValue::Object(&unscaled),
                    JValue::Int(i32::from(decimal_dtype.scale())),
                ],
            )?)
        }
        ScalarValue::Utf8(value) => Ok(env.new_string(value.as_str())?.into()),
        ScalarValue::Binary(value) => Ok(env.byte_array_from_slice(value.as_slice())?.into()),
        ScalarValue::Tuple(_) | ScalarValue::Union(_) | ScalarValue::Variant(_) => {
            Err(JNIError::Vortex(vortex_err!(
                "cannot return nested scalar statistic with dtype {} to Java",
                scalar.dtype()
            )))
        }
    }
}

/// The statistics of one top-level column that are returned to Java.
pub(crate) struct ColumnStatistics {
    null_count: Precision<Scalar>,
    nan_count: Precision<Scalar>,
    min: Precision<Scalar>,
    max: Precision<Scalar>,
}

impl ColumnStatistics {
    /// Reads the statistics from the aggregates that a writer recorded in a footer.
    fn from_aggregates(aggregates: Option<&AggregateStats>) -> Self {
        let get = |stat: Stat| {
            aggregates
                .zip(stat.aggregate_fn())
                .map_or(Precision::Absent, |(aggregates, aggregate_fn)| {
                    aggregates.get(&aggregate_fn)
                })
        };

        Self {
            null_count: get(Stat::NullCount),
            nan_count: get(Stat::NaNCount),
            min: get(Stat::Min),
            max: get(Stat::Max),
        }
    }

    /// Reads the statistics of the field at `path` from `source`.
    ///
    /// Stored aggregate partials come first. The legacy statistics are read only when a
    /// statistic has no partial, for example in a file written before partials were stored.
    async fn from_source(
        source: &dyn DataSource,
        path: &FieldPath,
        dtype: &DType,
    ) -> VortexResult<Self> {
        let mut legacy = None;

        Ok(Self {
            null_count: resolve_statistic(source, path, dtype, Stat::NullCount, &mut legacy)
                .await?,
            nan_count: resolve_statistic(source, path, dtype, Stat::NaNCount, &mut legacy).await?,
            min: resolve_statistic(source, path, dtype, Stat::Min, &mut legacy).await?,
            max: resolve_statistic(source, path, dtype, Stat::Max, &mut legacy).await?,
        })
    }
}

/// Resolves one statistic of the field at `path`, first from its stored aggregate partial and
/// then from the legacy statistics.
///
/// `legacy` caches the legacy statistics of the field, so they are read at most once.
async fn resolve_statistic(
    source: &dyn DataSource,
    path: &FieldPath,
    dtype: &DType,
    stat: Stat,
    legacy: &mut Option<StatsSet>,
) -> VortexResult<Precision<Scalar>> {
    let aggregate_fn = stat
        .aggregate_fn()
        .ok_or_else(|| vortex_err!("statistic {stat} has no aggregate function"))?;

    let partial = source.field_statistic(path, &aggregate_fn).await?;
    if !partial.is_absent() {
        return partial
            .map(|partial| finalize_partial(&aggregate_fn, dtype, &partial))
            .transpose();
    }

    if legacy.is_none() {
        *legacy = Some(source.field_statistics(path).await?);
    }

    let Some(stat_dtype) = stat.dtype(dtype) else {
        return Ok(Precision::Absent);
    };

    Ok(legacy.as_ref().map_or(Precision::Absent, |legacy| {
        legacy.get(stat).into_scalar(stat_dtype)
    }))
}

/// Computes the final value of `aggregate_fn` from a stored partial over a field of `dtype`.
fn finalize_partial(
    aggregate_fn: &AggregateFnRef,
    dtype: &DType,
    partial: &Scalar,
) -> VortexResult<Scalar> {
    let mut accumulator = aggregate_fn.accumulator(dtype)?;
    accumulator.combine_partials(partial)?;
    accumulator.final_scalar()
}

/// Returns the path and dtype of each top-level column in schema order.
///
/// A non-struct file has a single entry for its root column.
fn columns(footer: &Footer) -> Vec<(FieldPath, DType)> {
    match footer.dtype().as_struct_fields_opt() {
        Some(fields) => fields
            .names()
            .iter()
            .zip(fields.fields())
            .map(|(name, dtype)| (FieldPath::from_name(name.clone()), dtype))
            .collect(),
        None => vec![(FieldPath::root(), footer.dtype().clone())],
    }
}

/// Reads the statistics of each top-level column from the aggregates in `footer`.
///
/// Use this for a footer that a writer just produced, which has no legacy statistics to fall
/// back to.
pub(crate) fn footer_column_statistics(footer: &Footer) -> Vec<ColumnStatistics> {
    columns(footer)
        .iter()
        .map(|(path, _)| {
            let aggregates = footer
                .statistics()
                .and_then(|statistics| statistics.get_by_path(path))
                .map(|(aggregates, _)| aggregates);
            ColumnStatistics::from_aggregates(aggregates)
        })
        .collect()
}

/// Reads the statistics of each top-level column of the file that `footer` describes from
/// `source`, the data source of that file.
pub(crate) async fn source_column_statistics(
    source: &dyn DataSource,
    footer: &Footer,
) -> VortexResult<Vec<ColumnStatistics>> {
    let mut statistics = Vec::new();
    for (path, dtype) in columns(footer) {
        statistics.push(ColumnStatistics::from_source(source, &path, &dtype).await?);
    }

    Ok(statistics)
}

/// Returns the compressed size in bytes of each top-level column in schema order.
///
/// A non-struct file has a single entry for its root column.
fn compressed_column_sizes(footer: &Footer) -> VortexResult<Vec<u64>> {
    let sizes = footer.compressed_field_sizes()?;
    let Some(fields) = footer.dtype().as_struct_fields_opt() else {
        return Ok(vec![sizes.total()]);
    };

    Ok(fields
        .names()
        .iter()
        .map(|name| {
            sizes
                .get(&FieldPath::from_name(name.clone()))
                .unwrap_or_default()
        })
        .collect())
}

/// Build a `dev.vortex.api.VortexColumnStatistics[]` with one entry per top-level column of
/// the file that `footer` describes.
///
/// `statistics` holds one entry per top-level column in schema order. The compressed sizes come
/// from the footer, so this performs no IO.
pub(crate) fn column_statistics_to_java<'local>(
    env: &mut Env<'local>,
    footer: &Footer,
    statistics: Vec<ColumnStatistics>,
) -> Result<JObjectArray<'local>, JNIError> {
    let column_sizes = compressed_column_sizes(footer)?;
    if column_sizes.len() != statistics.len() {
        return Err(vortex_err!(
            "expected statistics for {} columns, got {}",
            column_sizes.len(),
            statistics.len()
        )
        .into());
    }

    let row_count = checked_jlong(footer.row_count(), "row count")?;
    let columns = env.new_object_array(
        i32::try_from(column_sizes.len())
            .map_err(|_| vortex_err!("column count exceeds Java array range"))?,
        jni::jni_str!("dev/vortex/api/VortexColumnStatistics"),
        JObject::null(),
    )?;

    for (column_index, (compressed_size, column_statistics)) in
        column_sizes.into_iter().zip(statistics).enumerate()
    {
        let ColumnStatistics {
            null_count,
            nan_count,
            min,
            max,
        } = column_statistics;
        let null_count = exact_count_jlong(null_count, "null count")?;
        let nan_count = exact_count_jlong(nan_count, "NaN count")?;

        let column = env.with_local_frame_returning_local::<_, JObject, JNIError>(16, |env| {
            let lower_bound = match min.into_inner() {
                Some(value) => scalar_to_java(env, value)?,
                None => JObject::null(),
            };
            let upper_bound = match max.into_inner() {
                Some(value) => scalar_to_java(env, value)?,
                None => JObject::null(),
            };
            Ok(env.new_object(
                jni::jni_str!("dev/vortex/api/VortexColumnStatistics"),
                jni::jni_sig!("(IJJJJLjava/lang/Object;Ljava/lang/Object;)V"),
                &[
                    JValue::Int(
                        i32::try_from(column_index)
                            .map_err(|_| vortex_err!("column index exceeds Java int range"))?,
                    ),
                    JValue::Long(checked_jlong(compressed_size, "compressed column size")?),
                    JValue::Long(row_count),
                    JValue::Long(null_count),
                    JValue::Long(nan_count),
                    JValue::Object(&lower_bound),
                    JValue::Object(&upper_bound),
                ],
            )?)
        })?;
        columns.set_element(env, column_index, &column)?;
        env.delete_local_ref(column);
    }

    Ok(columns)
}
