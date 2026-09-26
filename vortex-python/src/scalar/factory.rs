// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Itertools;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyBool;
use pyo3::types::PyBytes;
use pyo3::types::PyDict;
use pyo3::types::PyFloat;
use pyo3::types::PyInt;
use pyo3::types::PyList;
use pyo3::types::PyString;
use vortex::dtype::BigCast;
use vortex::dtype::DType;
use vortex::dtype::DecimalDType;
use vortex::dtype::FieldName;
use vortex::dtype::FieldNames;
use vortex::dtype::MAX_PRECISION;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::dtype::StructFields;
use vortex::dtype::extension::ExtDType;
use vortex::dtype::i256;
use vortex::extension::datetime::Date;
use vortex::extension::datetime::Time;
use vortex::extension::datetime::TimeUnit;
use vortex::extension::datetime::Timestamp;
use vortex::extension::uuid::Uuid;
use vortex::extension::uuid::UuidMetadata;
use vortex::scalar::DecimalValue;
use vortex::scalar::Scalar;
use vortex::scalar::ScalarValue;

use crate::dtype::PyDType;
use crate::error::PyVortexResult;
use crate::scalar::PyScalar;
use crate::scalar::bool;

/// Construct a Vortex scalar from a Python value.
///
/// Parameters
/// ----------
/// value : :class:`object`
///     The value to convert. Supported types are :obj:`None`, :class:`bool`, :class:`int`,
///     :class:`float`, :class:`str`, :class:`bytes`, :class:`list`, :class:`dict` (as a struct),
///     :class:`decimal.Decimal`, :class:`datetime.date`, :class:`datetime.datetime`,
///     :class:`datetime.time`, :class:`uuid.UUID`, and :class:`vortex.Scalar`.
/// dtype : :class:`vortex.DType`, optional
///     The data type of the result. When given, it also guides the conversion: the unit of a
///     date, time or timestamp type, the scale of a decimal type, and the element and field types
///     of list and struct types. Without it, :class:`int` becomes a 64-bit integer,
///     :class:`float` a 64-bit float, :class:`decimal.Decimal` a decimal with the value's own
///     precision and scale, dates use days, and times and timestamps use microseconds.
///
/// Returns
/// -------
/// :class:`vortex.Scalar`
///
/// Raises
/// ------
/// ValueError
///     If the value cannot be represented exactly, for example a decimal or timestamp that
///     would lose precision in the requested scale or unit, a non-finite decimal, or a
///     timezone that has no IANA name.
///
/// Notes
/// -----
/// Comparisons in expressions require both sides to have the same type, so a literal compared
/// against a column should match the column's type, for example
/// ``vx.scalar(value, dtype=vx.timestamp("ns"))`` for a nanosecond timestamp column.
///
/// Examples
/// --------
///
/// ```python
/// >>> import datetime
/// >>> import vortex as vx
/// >>> vx.scalar(datetime.date(1970, 1, 2)).as_py()
/// 1
/// >>> vx.scalar(datetime.time(0, 0, 1), dtype=vx.time("ms")).as_py()
/// 1000
/// ```
#[pyfunction(name = "scalar")]
#[pyo3(signature = (value, *, dtype=None))]
pub fn scalar<'py>(
    py: Python<'py>,
    value: Bound<'py, PyAny>,
    dtype: Option<PyDType>,
) -> PyResult<Bound<'py, PyScalar>> {
    PyScalar::init(
        py,
        scalar_helper(&value, dtype.as_ref().map(|dtype| dtype.inner()))?,
    )
}

pub fn scalar_helper(value: &Bound<'_, PyAny>, dtype: Option<&DType>) -> PyVortexResult<Scalar> {
    let scalar = scalar_helper_inner(value, dtype)?;

    // If a dtype was provided, attempt to  cast the scalar to that dtype.
    // This is a trivially cheap no-op if the scalar is already of the correct type.
    if let Some(dtype) = dtype {
        Ok(scalar.cast(dtype)?)
    } else {
        Ok(scalar)
    }
}

/// Attempts to convert the python object to a scalar, with a hint of the expected
/// dtype. It can assume that the scalar_helper function will perform a final cast to the correct
/// dtype if necessary.
fn scalar_helper_inner(value: &Bound<'_, PyAny>, dtype: Option<&DType>) -> PyResult<Scalar> {
    // If it's already a scalar, return it
    if let Ok(value) = value.cast::<PyScalar>() {
        return Ok(value.get().inner().clone());
    }

    // Otherwise, we start checking the known Python types.

    // None
    if value.is_none() {
        return Ok(Scalar::null(dtype.cloned().unwrap_or(DType::Null)));
    }

    // bool
    if let Ok(bool) = value.cast::<PyBool>() {
        return Ok(Scalar::bool(
            bool.extract::<bool>()?,
            Nullability::NonNullable,
        ));
    }

    // decimal
    if let Some(decimal_dtype) = dtype.and_then(|d| d.as_decimal_opt()) {
        if is_decimal(value)? {
            return Ok(decimal_scalar(value, Some(decimal_dtype))?);
        }
        let value = if let Ok(v) = value.extract::<i8>() {
            DecimalValue::I8(v)
        } else if let Ok(v) = value.extract::<i16>() {
            DecimalValue::I16(v)
        } else if let Ok(v) = value.extract::<i32>() {
            DecimalValue::I32(v)
        } else if let Ok(v) = value.extract::<i64>() {
            DecimalValue::I64(v)
        } else if let Ok(v) = value.extract::<i128>() {
            DecimalValue::I128(v)
        } else {
            return Err(PyValueError::new_err(
                "Value can't be represented as decimal",
            ));
        };
        return Ok(Scalar::decimal(
            value,
            *decimal_dtype,
            Nullability::NonNullable,
        ));
    }

    if let Ok(integer) = value.cast::<PyInt>() {
        return Ok(Scalar::primitive(
            integer.extract::<i64>()?,
            Nullability::NonNullable,
        ));
    }

    // float
    if let Ok(float) = value.cast::<PyFloat>() {
        return Ok(Scalar::primitive(
            float.extract::<f64>()?,
            Nullability::NonNullable,
        ));
    }

    // str
    if let Ok(string) = value.cast::<PyString>() {
        return Ok(Scalar::utf8(
            string.extract::<String>()?,
            Nullability::NonNullable,
        ));
    }

    // bytes
    if let Ok(bytes) = value.cast::<PyBytes>() {
        return Ok(Scalar::binary(
            bytes.extract::<Vec<u8>>()?,
            Nullability::NonNullable,
        ));
    }

    // dict
    if let Ok(dict) = value.cast::<PyDict>() {
        // Extract the field names from the dictionary keys
        let names: FieldNames = dict
            .keys()
            .iter()
            .map(|key| key.extract::<String>())
            .map_ok(FieldName::from)
            .collect::<PyResult<Vec<FieldName>>>()?
            .into();

        if let Some(DType::Struct(dtype, nullability)) = dtype {
            if names != dtype.names() {
                return Err(PyValueError::new_err(format!(
                    "Dictionary field names {:?} do not match target dtype names {:?}",
                    names,
                    dtype.names()
                )));
            }

            // Each field is cast to its dtype, since `Scalar::struct_` requires every child to
            // match it exactly.
            let children: Vec<Scalar> = dict
                .values()
                .into_iter()
                .zip(dtype.fields())
                .map(|(item, field_dtype)| {
                    scalar_helper(&item, Some(&field_dtype)).map_err(PyErr::from)
                })
                .try_collect()?;
            return Ok(Scalar::struct_(
                DType::Struct(dtype.clone(), *nullability),
                children,
            ));
        } else {
            let values: Vec<Scalar> = dict
                .values()
                .into_iter()
                .map(|value| scalar_helper_inner(&value, None))
                .try_collect()?;
            let dtype = DType::Struct(
                StructFields::new(
                    names,
                    values.iter().map(|value| value.dtype().clone()).collect(),
                ),
                Nullability::NonNullable,
            );
            return Ok(Scalar::struct_(dtype, values));
        };
    }

    if let Ok(list) = value.cast::<PyList>() {
        if let Some(DType::List(element_dtype, ..)) = dtype {
            // Each element is cast to the element dtype, since `Scalar::list` requires every
            // child to match it exactly.
            let elements = list
                .iter()
                .map(|e| scalar_helper(&e, Some(element_dtype)).map_err(PyErr::from))
                .try_collect()?;
            return Ok(Scalar::list(
                Arc::clone(element_dtype),
                elements,
                Nullability::NonNullable,
            ));
        } else {
            // If no dtype was provided, we need to infer the element dtype from the list contents.
            // We do this in a greedy way taking the first element dtype we find.
            let mut elements = Vec::with_capacity(list.len());
            let mut element_dtype = None;

            for element in list.iter() {
                let scalar = scalar_helper_inner(&element, element_dtype.as_ref())?;
                if element_dtype.is_none() {
                    element_dtype = Some(scalar.dtype().clone());
                }
                elements.push(scalar);
            }

            return Ok(Scalar::list(
                element_dtype
                    .map(Arc::new)
                    // Empty list defaults to Null dtype
                    .unwrap_or_else(|| Arc::new(DType::Null)),
                elements,
                Nullability::NonNullable,
            ));
        }
    }

    // Standard library types, checked after the built-in types to keep those conversions cheap.
    let py = value.py();

    // decimal.Decimal without a decimal dtype hint
    if is_decimal(value)? {
        return Ok(decimal_scalar(value, None)?);
    }

    // datetime.datetime, checked before datetime.date because it is a subclass of it.
    let datetime_module = py.import(intern!(py, "datetime"))?;
    if value.is_instance(&datetime_module.getattr(intern!(py, "datetime"))?)? {
        return Ok(timestamp_scalar(value, dtype)?);
    }

    // datetime.date
    if value.is_instance(&datetime_module.getattr(intern!(py, "date"))?)? {
        return Ok(date_scalar(value, dtype)?);
    }

    // datetime.time
    if value.is_instance(&datetime_module.getattr(intern!(py, "time"))?)? {
        return Ok(time_scalar(value, dtype)?);
    }

    // uuid.UUID
    let uuid_type = py
        .import(intern!(py, "uuid"))?
        .getattr(intern!(py, "UUID"))?;
    if value.is_instance(&uuid_type)? {
        let bytes: Vec<u8> = value.getattr(intern!(py, "bytes"))?.extract()?;
        return Ok(uuid_scalar(&bytes)?);
    }

    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "Cannot convert Python object to Vortex scalar: {}",
        value.get_type()
    )))
}

/// Proleptic Gregorian ordinal of 1970-01-01, as returned by `date.toordinal()`.
const UNIX_EPOCH_ORDINAL: i64 = 719_163;
const MICROS_PER_SECOND: i64 = 1_000_000;
const MICROS_PER_DAY: i64 = 86_400 * MICROS_PER_SECOND;
/// Number of bytes in a UUID.
const UUID_BYTE_LEN: usize = 16;

fn is_decimal(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = value.py();
    let decimal_type = py
        .import(intern!(py, "decimal"))?
        .getattr(intern!(py, "Decimal"))?;
    value.is_instance(&decimal_type)
}

/// Convert a count of microseconds into `unit`, failing rather than silently truncating.
fn micros_to_unit(micros: i64, unit: TimeUnit) -> PyResult<i64> {
    let per_unit = match unit {
        TimeUnit::Nanoseconds => {
            return micros.checked_mul(1_000).ok_or_else(|| {
                PyValueError::new_err(format!("{micros}us overflows i64 nanoseconds"))
            });
        }
        TimeUnit::Microseconds => return Ok(micros),
        TimeUnit::Milliseconds => 1_000,
        TimeUnit::Seconds => MICROS_PER_SECOND,
        TimeUnit::Days => MICROS_PER_DAY,
    };
    if micros % per_unit != 0 {
        return Err(PyValueError::new_err(format!(
            "{micros}us cannot be represented in {unit} without losing precision"
        )));
    }
    Ok(micros / per_unit)
}

/// Microseconds since midnight of a `datetime.time` or `datetime.datetime`.
fn time_of_day_micros(value: &Bound<'_, PyAny>) -> PyResult<i64> {
    let py = value.py();
    let hour: i64 = value.getattr(intern!(py, "hour"))?.extract()?;
    let minute: i64 = value.getattr(intern!(py, "minute"))?.extract()?;
    let second: i64 = value.getattr(intern!(py, "second"))?.extract()?;
    let microsecond: i64 = value.getattr(intern!(py, "microsecond"))?.extract()?;
    Ok(((hour * 60 + minute) * 60 + second) * MICROS_PER_SECOND + microsecond)
}

/// Days since the Unix epoch of a `datetime.date` or `datetime.datetime`.
fn epoch_days(value: &Bound<'_, PyAny>) -> PyResult<i64> {
    let ordinal: i64 = value
        .call_method0(intern!(value.py(), "toordinal"))?
        .extract()?;
    Ok(ordinal - UNIX_EPOCH_ORDINAL)
}

/// The Vortex timezone name of a `tzinfo`.
///
/// Vortex timestamps carry an IANA zone name, so only `zoneinfo.ZoneInfo` and pytz zones and
/// `datetime.timezone.utc` are accepted; other fixed-offset zones have no IANA name.
fn timezone_name(tzinfo: &Bound<'_, PyAny>) -> PyResult<String> {
    let py = tzinfo.py();
    // `zoneinfo.ZoneInfo` names its zone `key`; pytz zones name it `zone`.
    for attr in [intern!(py, "key"), intern!(py, "zone")] {
        if let Ok(name) = tzinfo.getattr(attr)
            && let Ok(Some(name)) = name.extract::<Option<String>>()
        {
            return Ok(name);
        }
    }
    let utc = py
        .import(intern!(py, "datetime"))?
        .getattr(intern!(py, "timezone"))?
        .getattr(intern!(py, "utc"))?;
    if tzinfo.eq(utc)? {
        return Ok("UTC".to_string());
    }
    Err(PyValueError::new_err(format!(
        "Unsupported timezone {}: use a zoneinfo.ZoneInfo, a pytz zone or datetime.timezone.utc",
        tzinfo.repr()?
    )))
}

/// Convert a `datetime.date` into a Vortex `Date` scalar.
///
/// The unit is taken from `dtype` when it is a `Date` dtype, and otherwise defaults to days.
fn date_scalar(value: &Bound<'_, PyAny>, dtype: Option<&DType>) -> PyVortexResult<Scalar> {
    let unit = dtype
        .and_then(DType::as_extension_opt)
        .and_then(|ext| ext.metadata_opt::<Date>())
        .copied()
        .unwrap_or(TimeUnit::Days);
    let ext = Date::try_new(unit, Nullability::NonNullable)?;
    let days = epoch_days(value)?;
    let storage = match unit {
        TimeUnit::Days => ScalarValue::from(
            i32::try_from(days)
                .map_err(|_| PyValueError::new_err(format!("{days} days does not fit in i32")))?,
        ),
        _ => ScalarValue::from(micros_to_unit(days * MICROS_PER_DAY, unit)?),
    };
    Ok(Scalar::try_new(
        DType::Extension(ext.erased()),
        Some(storage),
    )?)
}

/// Convert a `datetime.datetime` into a Vortex `Timestamp` scalar.
///
/// The unit and timezone are taken from `dtype` when it is a `Timestamp` dtype, and otherwise
/// default to microseconds and the value's own timezone. Naive values are stored as their
/// wall-clock time; timezone-aware values are stored as the UTC instant. A naive value cannot
/// be converted to a timezone-aware dtype, nor an aware value to a naive dtype.
fn timestamp_scalar(value: &Bound<'_, PyAny>, dtype: Option<&DType>) -> PyVortexResult<Scalar> {
    let py = value.py();
    let tzinfo = value.getattr(intern!(py, "tzinfo"))?;
    let aware = !tzinfo.is_none();

    let mut micros = epoch_days(value)? * MICROS_PER_DAY + time_of_day_micros(value)?;
    if aware {
        let offset = value.call_method0(intern!(py, "utcoffset"))?;
        let days: i64 = offset.getattr(intern!(py, "days"))?.extract()?;
        let seconds: i64 = offset.getattr(intern!(py, "seconds"))?.extract()?;
        let microseconds: i64 = offset.getattr(intern!(py, "microseconds"))?.extract()?;
        micros -= days * MICROS_PER_DAY + seconds * MICROS_PER_SECOND + microseconds;
    }

    let options = dtype
        .and_then(DType::as_extension_opt)
        .and_then(|ext| ext.metadata_opt::<Timestamp>());
    let (unit, tz) = match options {
        Some(options) => {
            if options.tz.is_some() != aware {
                return Err(PyValueError::new_err(format!(
                    "Cannot convert a {} datetime to a timestamp dtype {} a timezone",
                    if aware { "timezone-aware" } else { "naive" },
                    if aware { "without" } else { "with" },
                ))
                .into());
            }
            (options.unit, options.tz.clone())
        }
        None => {
            let tz = if aware {
                Some(Arc::from(timezone_name(&tzinfo)?.as_str()))
            } else {
                None
            };
            (TimeUnit::Microseconds, tz)
        }
    };

    // `pandas.Timestamp` subclasses `datetime.datetime` and keeps its sub-microsecond part in
    // `nanosecond`, which `microsecond` excludes.
    let nanosecond: i64 = match value.getattr(intern!(py, "nanosecond")) {
        Ok(nanosecond) => nanosecond.extract()?,
        Err(_) => 0,
    };
    let storage = if unit == TimeUnit::Nanoseconds {
        micros_to_unit(micros, unit)?
            .checked_add(nanosecond)
            .ok_or_else(|| PyValueError::new_err("Timestamp overflows i64 nanoseconds"))?
    } else if nanosecond != 0 {
        return Err(PyValueError::new_err(format!(
            "Timestamp with {nanosecond}ns cannot be represented in {unit} without losing \
             precision; pass dtype=vx.timestamp(\"ns\")"
        ))
        .into());
    } else {
        micros_to_unit(micros, unit)?
    };

    let ext = Timestamp::new_with_tz(unit, tz, Nullability::NonNullable);
    Ok(Scalar::try_new(
        DType::Extension(ext.erased()),
        Some(ScalarValue::from(storage)),
    )?)
}

/// Convert a `decimal.Decimal` into a Vortex decimal scalar.
///
/// With a decimal dtype hint the value is rescaled to its scale exactly; otherwise the precision
/// and scale are inferred from the value's digits and exponent. Non-finite values, values with
/// more than [`MAX_PRECISION`] digits, and rescaling that would drop non-zero digits are errors.
fn decimal_scalar(value: &Bound<'_, PyAny>, hint: Option<&DecimalDType>) -> PyVortexResult<Scalar> {
    let py = value.py();
    let parts = value.call_method0(intern!(py, "as_tuple"))?;
    // The exponent is a string ('n', 'N' or 'F') for NaN and infinities.
    let Ok(exponent) = parts.getattr(intern!(py, "exponent"))?.extract::<i64>() else {
        return Err(PyValueError::new_err(format!(
            "Cannot convert non-finite decimal {} to a Vortex scalar",
            value.str()?
        ))
        .into());
    };
    let negative = parts.getattr(intern!(py, "sign"))?.extract::<u8>()? == 1;
    let digits: Vec<u8> = parts.getattr(intern!(py, "digits"))?.extract()?;
    let digits = match digits.iter().position(|&d| d != 0) {
        Some(first) => &digits[first..],
        None => &[][..],
    };

    let scale: i64 = hint.map_or_else(|| (-exponent).max(0), |d| d.scale().into());
    // Shift the digits so that they represent the unscaled value at `scale`.
    let shift = exponent + scale;
    let unscaled_digits: Vec<u8> = if digits.is_empty() {
        Vec::new()
    } else if let Ok(zeros) = usize::try_from(shift) {
        if digits.len().saturating_add(zeros) > usize::from(MAX_PRECISION) {
            return Err(PyValueError::new_err(format!(
                "Decimal {} has more than {MAX_PRECISION} digits at scale {scale}",
                value.str()?
            ))
            .into());
        }
        let mut shifted = digits.to_vec();
        shifted.resize(digits.len() + zeros, 0);
        shifted
    } else {
        let keep = digits
            .len()
            .saturating_sub(usize::try_from(-shift).unwrap_or(usize::MAX));
        if digits[keep..].iter().any(|&d| d != 0) {
            return Err(PyValueError::new_err(format!(
                "Decimal {} cannot be represented with scale {scale} without losing precision",
                value.str()?
            ))
            .into());
        }
        digits[..keep].to_vec()
    };

    let decimal_dtype = match hint {
        Some(hint) => {
            if unscaled_digits.len() > usize::from(hint.precision()) {
                return Err(PyValueError::new_err(format!(
                    "Decimal {} does not fit in precision {}",
                    value.str()?,
                    hint.precision()
                ))
                .into());
            }
            *hint
        }
        None => {
            let precision = i64::try_from(unscaled_digits.len())
                .unwrap_or(i64::MAX)
                .max(scale)
                .max(1);
            DecimalDType::try_new(
                u8::try_from(precision).unwrap_or(u8::MAX),
                i8::try_from(scale).unwrap_or(i8::MAX),
            )
            .map_err(|err| {
                PyValueError::new_err(format!(
                    "Decimal {} cannot be represented as a Vortex decimal: {err}",
                    value
                ))
            })?
        }
    };

    let ten = i256::from_i128(10);
    let mut unscaled = i256::ZERO;
    for digit in unscaled_digits {
        unscaled = unscaled * ten + i256::from_i128(digit.into());
    }
    if negative {
        unscaled = -unscaled;
    }

    Ok(Scalar::try_new(
        DType::Decimal(decimal_dtype, Nullability::NonNullable),
        Some(ScalarValue::from(narrowest_decimal_value(
            unscaled,
            &decimal_dtype,
        )?)),
    )?)
}

/// The narrowest [`DecimalValue`] variant able to hold every value of `dtype`'s precision.
fn narrowest_decimal_value(value: i256, dtype: &DecimalDType) -> PyResult<DecimalValue> {
    let overflow = || PyValueError::new_err(format!("Decimal value {value} overflows {dtype}"));
    let required_bits = dtype.required_bit_width();
    Ok(if required_bits <= 8 {
        DecimalValue::I8(BigCast::from(value).ok_or_else(overflow)?)
    } else if required_bits <= 16 {
        DecimalValue::I16(BigCast::from(value).ok_or_else(overflow)?)
    } else if required_bits <= 32 {
        DecimalValue::I32(BigCast::from(value).ok_or_else(overflow)?)
    } else if required_bits <= 64 {
        DecimalValue::I64(BigCast::from(value).ok_or_else(overflow)?)
    } else if required_bits <= 128 {
        DecimalValue::I128(value.maybe_i128().ok_or_else(overflow)?)
    } else {
        DecimalValue::I256(value)
    })
}

/// Build a UUID extension scalar from its 16 big-endian bytes, matching `uuid.UUID.bytes`.
fn uuid_scalar(bytes: &[u8]) -> PyVortexResult<Scalar> {
    let list_size = u32::try_from(UUID_BYTE_LEN).map_err(|_| {
        PyValueError::new_err(format!(
            "UUID byte length {UUID_BYTE_LEN} does not fit in u32"
        ))
    })?;
    if bytes.len() != UUID_BYTE_LEN {
        return Err(PyValueError::new_err(format!(
            "UUID must be exactly {UUID_BYTE_LEN} bytes, got {}",
            bytes.len()
        ))
        .into());
    }
    let element_dtype = DType::Primitive(PType::U8, Nullability::NonNullable);
    let storage = Scalar::fixed_size_list(
        element_dtype.clone(),
        bytes
            .iter()
            .map(|&b| Scalar::primitive(b, Nullability::NonNullable))
            .collect(),
        Nullability::NonNullable,
    );
    let ext = ExtDType::<Uuid>::try_new(
        UuidMetadata::default(),
        DType::FixedSizeList(Arc::new(element_dtype), list_size, Nullability::NonNullable),
    )?;
    Ok(Scalar::try_new(
        DType::Extension(ext.erased()),
        storage.into_value(),
    )?)
}

/// Convert a naive `datetime.time` into a Vortex `Time` scalar.
///
/// The unit is taken from `dtype` when it is a `Time` dtype, and otherwise defaults to
/// microseconds, the resolution of `datetime.time`. Converting to a coarser unit that would drop
/// a non-zero fraction of a second is an error rather than a silent truncation.
fn time_scalar(value: &Bound<'_, PyAny>, dtype: Option<&DType>) -> PyVortexResult<Scalar> {
    let py = value.py();
    if !value.getattr(intern!(py, "tzinfo"))?.is_none() {
        return Err(PyValueError::new_err(
            "Timezone-aware datetime.time values cannot be converted to a Vortex time scalar",
        )
        .into());
    }
    let unit = dtype
        .and_then(DType::as_extension_opt)
        .and_then(|ext| ext.metadata_opt::<Time>())
        .copied()
        .unwrap_or(TimeUnit::Microseconds);
    let ext = Time::try_new(unit, Nullability::NonNullable)?;
    let value = micros_to_unit(time_of_day_micros(value)?, unit)?;
    let storage = match unit {
        TimeUnit::Seconds | TimeUnit::Milliseconds => {
            ScalarValue::from(i32::try_from(value).map_err(|_| {
                PyValueError::new_err(format!("Time value does not fit in i32 {unit}"))
            })?)
        }
        _ => ScalarValue::from(value),
    };
    Ok(Scalar::try_new(
        DType::Extension(ext.erased()),
        Some(storage),
    )?)
}
