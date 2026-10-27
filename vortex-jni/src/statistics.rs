// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversion of the per-column statistics stored in a Vortex file footer into
//! `dev.vortex.api.VortexColumnStatistics` objects.

use jni::Env;
use jni::objects::JObject;
use jni::objects::JObjectArray;
use jni::objects::JValue;
use jni::sys::JNI_FALSE;
use jni::sys::JNI_TRUE;
use jni::sys::jlong;
use vortex::aggregate_fn::AggregateFnRef;
use vortex::aggregate_fn::AggregateFnVTableExt;
use vortex::aggregate_fn::EmptyOptions;
use vortex::aggregate_fn::NumericalAggregateOpts;
use vortex::aggregate_fn::fns::max::Max;
use vortex::aggregate_fn::fns::min::Min;
use vortex::aggregate_fn::fns::nan_count::NanCount;
use vortex::aggregate_fn::fns::null_count::NullCount;
use vortex::array::scalar::PValue;
use vortex::array::scalar::Scalar;
use vortex::array::scalar::ScalarValue;
use vortex::dtype::DType;
use vortex::dtype::FieldPath;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::file::Footer;
use vortex::layout::layouts::file_stats::AggregateStats;

use crate::errors::JNIError;

pub(crate) fn checked_jlong(value: u64, name: &str) -> VortexResult<jlong> {
    jlong::try_from(value).map_err(|_| vortex_err!("{name} exceeds Java long range: {value}"))
}

fn exact_count_jlong(
    aggregates: Option<&AggregateStats>,
    aggregate_fn: &AggregateFnRef,
) -> VortexResult<jlong> {
    aggregates
        .and_then(|aggregates| aggregates.get(aggregate_fn).as_exact())
        .and_then(|count| u64::try_from(&count).ok())
        .map(|value| checked_jlong(value, &aggregate_fn.to_string()))
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
/// All values come from the footer, so this performs no IO.
pub(crate) fn column_statistics_to_java<'local>(
    env: &mut Env<'local>,
    footer: &Footer,
) -> Result<JObjectArray<'local>, JNIError> {
    let column_sizes = compressed_column_sizes(footer)?;
    let row_count = checked_jlong(footer.row_count(), "row count")?;
    let file_stats = footer.statistics();
    let struct_fields = footer.dtype().as_struct_fields_opt();
    let columns = env.new_object_array(
        i32::try_from(column_sizes.len())
            .map_err(|_| vortex_err!("column count exceeds Java array range"))?,
        jni::jni_str!("dev/vortex/api/VortexColumnStatistics"),
        JObject::null(),
    )?;

    for (column_index, compressed_size) in column_sizes.into_iter().enumerate() {
        let aggregates = struct_fields
            .and_then(|fields| fields.names().get(column_index))
            .zip(file_stats)
            .and_then(|(name, all_stats)| {
                all_stats.get_by_path(&FieldPath::from_name(name.clone()))
            })
            .map(|(aggregates, _)| aggregates);
        let null_count = exact_count_jlong(aggregates, &NullCount.bind(EmptyOptions))?;
        let nan_count = exact_count_jlong(aggregates, &NanCount.bind(EmptyOptions))?;
        let lower_bound = aggregates.and_then(|aggregates| {
            aggregates
                .get(&Min.bind(NumericalAggregateOpts::skip_nans()))
                .into_inner()
        });
        let upper_bound = aggregates.and_then(|aggregates| {
            aggregates
                .get(&Max.bind(NumericalAggregateOpts::skip_nans()))
                .into_inner()
        });

        let column = env.with_local_frame_returning_local::<_, JObject, JNIError>(16, |env| {
            let lower_bound = match lower_bound {
                Some(value) => scalar_to_java(env, value)?,
                None => JObject::null(),
            };
            let upper_bound = match upper_bound {
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
