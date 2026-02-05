// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use pyo3::Bound;
use pyo3::PyResult;
use pyo3::Python;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::PyAnyMethods;
use pyo3::pyfunction;
use pyo3::types::PyDict;
use vortex::dtype::DType;
use vortex::dtype::DecimalDType;
use vortex::dtype::FieldName;
use vortex::dtype::FieldNames;
use vortex::dtype::PType;
use vortex::dtype::StructFields;
use vortex::dtype::datetime::Date;
use vortex::dtype::datetime::Time;
use vortex::dtype::datetime::TimeUnit;
use vortex::dtype::datetime::Timestamp;

use crate::dtype::PyDType;
use crate::error::PyVortexResult;

/// Construct the data type for a column containing only the null value.
///
/// Returns
/// -------
/// :class:`vortex.DType`
///
/// Examples
/// --------
///
/// A data type permitting only :obj:`None`.
///
/// ```python
/// >>> vx.null()
/// null()
/// ```
#[pyfunction(name = "null")]
#[pyo3(signature = ())]
pub(super) fn dtype_null(py: Python<'_>) -> PyResult<Bound<'_, PyDType>> {
    PyDType::init(py, DType::Null)
}

/// Construct a Boolean data type.
///
/// Parameters
/// ----------
/// nullable : :class:`bool`
///     When :obj:`True`, :obj:`None` is a permissible value.
///
/// Returns
/// -------
/// :class:`vortex.DType`
///
/// Examples
/// --------
///
/// A data type permitting :obj:`None`, :obj:`True`, and :obj:`False`.
///
/// ```python
/// >>> import vortex as vx
/// >>> vx.bool_(nullable=True)
/// bool(nullable=True)
/// ```
///
/// A data type permitting just :obj:`True` and :obj:`False`.
///
/// ```python
/// >>> vx.bool_()
/// bool(nullable=False)
/// ```
#[pyfunction(name = "bool_")]
#[pyo3(signature = (*, nullable = false))]
pub(super) fn dtype_bool(py: Python<'_>, nullable: bool) -> PyResult<Bound<'_, PyDType>> {
    PyDType::init(py, DType::Bool(nullable.into()))
}

/// Construct a signed integral data type.
///
/// Parameters
/// ----------
/// width : Literal[8, 16, 32, 64].
///     The bit width determines the span of valid values. If :obj:`None`, 64 is used.
///
/// nullable : :class:`bool`
///     When :obj:`True`, :obj:`None` is a permissible value.
///
/// Returns
/// -------
/// :class:`vortex.DType`
///
/// Examples
/// --------
///
/// A data type permitting :obj:`None` and the integers from -128 to 127, inclusive:
///
/// ```python
/// >>> import vortex as vx
/// >>> vx.int_(8, nullable=True)
/// int(8, nullable=True)
/// ```
///
/// A data type permitting just the integers from -2,147,483,648 to 2,147,483,647, inclusive:
///
/// ```python
/// >>> vx.int_(32)
/// int(32, nullable=False)
/// ```
#[pyfunction(name = "int_")]
#[pyo3(signature = (width = 64, *, nullable = false))]
pub(super) fn dtype_int(
    py: Python<'_>,
    width: u16,
    nullable: bool,
) -> PyResult<Bound<'_, PyDType>> {
    let dtype = match width {
        8 => DType::Primitive(PType::I8, nullable.into()),
        16 => DType::Primitive(PType::I16, nullable.into()),
        32 => DType::Primitive(PType::I32, nullable.into()),
        64 => DType::Primitive(PType::I64, nullable.into()),
        _ => return Err(PyValueError::new_err("Invalid int width")),
    };
    PyDType::init(py, dtype)
}

/// Construct an unsigned integral data type.
///
/// Parameters
/// ----------
/// width : Literal[8, 16, 32, 64].
///     The bit width determines the span of valid values. If :obj:`None`, 64 is used.
///
/// nullable : :class:`bool`
///     When :obj:`True`, :obj:`None` is a permissible value.
///
/// Returns
/// -------
/// :class:`vortex.DType`
///
/// Examples
/// --------
///
/// A data type permitting :obj:`None` and the integers from 0 to 255, inclusive:
///
/// ```python
/// >>> import vortex as vx
/// >>> vx.uint(8, nullable=True)
/// uint(8, nullable=True)
/// ```
///
/// A data type permitting just the integers from 0 to 4,294,967,296 inclusive:
///
/// ```python
/// >>> vx.uint(32, nullable=False)
/// uint(32, nullable=False)
/// ```
#[pyfunction(name = "uint")]
#[pyo3(signature = (width = 64, *, nullable = false))]
pub(super) fn dtype_uint(
    py: Python<'_>,
    width: u16,
    nullable: bool,
) -> PyResult<Bound<'_, PyDType>> {
    let dtype = match width {
        8 => DType::Primitive(PType::U8, nullable.into()),
        16 => DType::Primitive(PType::U16, nullable.into()),
        32 => DType::Primitive(PType::U32, nullable.into()),
        64 => DType::Primitive(PType::U64, nullable.into()),
        _ => return Err(PyValueError::new_err("Invalid uint width")),
    };
    PyDType::init(py, dtype)
}

/// Construct an IEEE 754 binary floating-point data type.
///
/// Parameters
/// ----------
/// width : Literal[16, 32, 64].
///     The bit width determines the range and precision of the floating-point values. If
///     :obj:`None`, 64 is used.
///
/// nullable : :class:`bool`
///     When :obj:`True`, :obj:`None` is a permissible value.
///
/// Returns
/// -------
/// :class:`vortex.DType`
///
/// Examples
/// --------
///
/// A data type permitting :obj:`None` as well as IEEE 754 binary16 floating-point values. Values
/// larger than 65,520 or less than -65,520 will respectively round to positive and negative
/// infinity.
///
/// ```python
/// >>> import vortex as vx
/// >>> vx.float_(16, nullable=False)
/// float(16, nullable=False)
/// ```
#[pyfunction(name = "float_")]
#[pyo3(signature = (width = 64, *, nullable = false))]
pub(super) fn dtype_float(
    py: Python<'_>,
    width: i8,
    nullable: bool,
) -> PyResult<Bound<'_, PyDType>> {
    let dtype = match width {
        16 => DType::Primitive(PType::F16, nullable.into()),
        32 => DType::Primitive(PType::F32, nullable.into()),
        64 => DType::Primitive(PType::F64, nullable.into()),
        _ => return Err(PyValueError::new_err("Invalid float width")),
    };
    PyDType::init(py, dtype)
}

/// Construct a decimal data type.
///
/// Parameters
/// ----------
/// precision : :class:`int`
///     The number of significant digits representable by an instance of this type.
///
/// scale : :class:`int`
///     The number of digits after the decimal point that are represented. If negative, the
///     number of trailing zeros in the whole number portion.
///
/// nullable : :class:`bool`
///     When :obj:`True`, :obj:`None` is a permissible value.
///
/// Returns
/// -------
/// :class:`vortex.DType`
///
/// Examples
/// --------
///
/// A data type permitting :obj:`None` and the integers from -128 to 127, inclusive:
///
/// ```python
/// >>> import vortex as vx
/// >>> vx.decimal(precision=13, scale=2, nullable=True)
/// decimal(precision=13, scale=2, nullable=True)
/// ```
///
/// A data type representing fixed-width decimal numbers with `precision` significant figures and
/// `scale` digits after the decimal point. If `scale` is a negative value, then it is taken
/// to be the number of trailing zeros before the decimal point.
///
/// ```python
/// >>> vx.decimal(precision = 10, scale = 3)
/// decimal(precision=10, scale=3, nullable=False)
/// ```
#[pyfunction(name = "decimal")]
#[pyo3(signature = (*, precision = 10, scale = 0, nullable = false))]
pub(super) fn dtype_decimal(
    py: Python<'_>,
    precision: u8,
    scale: i8,
    nullable: bool,
) -> PyVortexResult<Bound<'_, PyDType>> {
    let decimal_type = DType::Decimal(DecimalDType::try_new(precision, scale)?, nullable.into());
    Ok(PyDType::init(py, decimal_type)?)
}

/// Construct a UTF-8-encoded string data type.
///
/// Parameters
/// ----------
/// nullable : :class:`bool`
///     When :obj:`True`, :obj:`None` is a permissible value.
///
/// Returns
/// -------
/// :class:`vortex.DType`
///
/// Examples
/// ---------
///
/// A data type permitting any UTF-8-encoded string, such as :code:`"Hello World"`, but not
/// permitting :obj:`None`.
///
/// ```python
/// >>> import vortex as vx
/// >>> vx.utf8(nullable=False)
/// utf8(nullable=False)
/// ```
#[pyfunction(name = "utf8")]
#[pyo3(signature = (*, nullable = false))]
pub(super) fn dtype_utf8(py: Python<'_>, nullable: bool) -> PyResult<Bound<'_, PyDType>> {
    PyDType::init(py, DType::Utf8(nullable.into()))
}

/// Construct a binary data type.
///
/// Parameters
/// ----------
/// nullable : :class:`bool`
///     When :obj:`True`, :obj:`None` is a permissible value.
///
/// Returns
/// -------
/// :class:`vortex.DType`
///
/// Examples
/// --------
///
/// A data type permitting any string of bytes but not permitting :obj:`None`.
///
/// ```python
/// >>> import vortex as vx
/// >>> vx.binary(nullable=False)
/// binary(nullable=False)
/// ```
#[pyfunction(name = "binary")]
#[pyo3(signature = (*, nullable = false))]
pub(super) fn dtype_binary(py: Python<'_>, nullable: bool) -> PyResult<Bound<'_, PyDType>> {
    PyDType::init(py, DType::Binary(nullable.into()))
}

/// Construct a struct data type.
///
/// Parameters
/// ----------
/// fields : :class:`dict`
///     A mapping from field names to data types.
/// nullable : :class:`bool`
///     When :obj:`True`, :obj:`None` is a permissible value.
///
/// Returns
/// -------
/// :class:`vortex.DType`
///
/// Examples
/// --------
///
/// A data type permitting a struct with two fields, :code:`"name"` and :code:`"age"`, where :code:`"name"` is a UTF-8-encoded string and :code:`"age"` is a 32-bit signed integer:
///
/// ```python
/// >>> import vortex as vx
/// >>> vx.struct({"name": vx.utf8(), "age": vx.int_(32)})
/// struct({"name": utf8(nullable=False), "age": int(32, nullable=False)}, nullable=False)
/// ```
// TODO(ngates): return a StructDType to allow inspection of fields
#[pyfunction(name = "struct")]
#[pyo3(signature = (fields = None, *, nullable = false))]
pub(super) fn dtype_struct<'py>(
    py: Python<'py>,
    fields: Option<&'py Bound<'py, PyDict>>,
    nullable: bool,
) -> PyResult<Bound<'py, PyDType>> {
    if let Some(fields) = fields {
        let nfields = fields.len()?;
        let mut names = Vec::with_capacity(nfields);
        let mut dtypes = Vec::with_capacity(nfields);

        for (name, field) in fields.into_iter() {
            let field_name = FieldName::from(name.to_string());
            let field_dtype: PyDType = field.extract()?;
            names.push(field_name);
            dtypes.push(field_dtype.inner().clone());
        }

        PyDType::init(
            py,
            DType::Struct(StructFields::new(names.into(), dtypes), nullable.into()),
        )
    } else {
        PyDType::init(
            py,
            DType::Struct(
                StructFields::new(FieldNames::empty(), vec![]),
                nullable.into(),
            ),
        )
    }
}

/// Construct a list data type.
///
/// Parameters
/// ----------
/// element : :class:`DType`
///     The type of the list element.
/// nullable : :class:`bool`
///     When :obj:`True`, :obj:`None` is a permissible value (note that this is the nullability of
///     the lists, not the inner elements).
///
/// Returns
/// -------
/// :class:`vortex.DType`
///
/// Examples
/// --------
///
/// A data type permitting a list of 32-bit signed integers, but not permitting :obj:`None`.
///
/// ```python
/// >>> import vortex as vx
/// >>> vx.list_(vx.int_(32), nullable=False)
/// list(int(32, nullable=False), nullable=False)
/// ```
#[pyfunction(name = "list_")]
#[pyo3(signature = (element, *, nullable = false))]
pub(super) fn dtype_list<'py>(
    element: &'py Bound<'py, PyDType>,
    nullable: bool,
) -> PyResult<Bound<'py, PyDType>> {
    PyDType::init(
        element.py(),
        DType::List(Arc::new(element.get().inner().clone()), nullable.into()),
    )
}

/// Construct a fixed-size list data type.
///
/// Parameters
/// ----------
/// element : :class:`DType`
///     The element type of the fixed-size list.
/// size : :class:`int`
///     The size of each list scalar.
/// nullable : :class:`bool`
///     When :obj:`True`, :obj:`None` is a permissible value (note that this is the nullability of
///     the fixed-size lists, not the inner elements).
///
/// Returns
/// -------
/// :class:`DType`
///
/// Examples
/// --------
///
/// Create a fixed-size list of 3 integers:
///
/// ```python
/// >>> import vortex as vx
/// >>> vx.fixed_size_list(vx.int_(32), 3, nullable=False)
/// fixed_size_list(int(32, nullable=False), 3, nullable=False)
/// ```
#[pyfunction(name = "fixed_size_list")]
#[pyo3(signature = (element, size, *, nullable = false))]
pub(super) fn dtype_fixed_size_list<'py>(
    element: &'py Bound<'py, PyDType>,
    size: u32,
    nullable: bool,
) -> PyResult<Bound<'py, PyDType>> {
    PyDType::init(
        element.py(),
        DType::FixedSizeList(
            Arc::new(element.get().inner().clone()),
            size,
            nullable.into(),
        ),
    )
}

fn parse_time_unit(unit: &str) -> PyResult<TimeUnit> {
    match unit {
        "ns" => Ok(TimeUnit::Nanoseconds),
        "us" => Ok(TimeUnit::Microseconds),
        "ms" => Ok(TimeUnit::Milliseconds),
        "s" => Ok(TimeUnit::Seconds),
        "days" => Ok(TimeUnit::Days),
        _ => Err(PyValueError::new_err(format!(
            "Invalid time unit: '{unit}'. Expected one of: 'ns', 'us', 'ms', 's', 'days'"
        ))),
    }
}

/// Construct a date data type.
///
/// Parameters
/// ----------
/// unit : :class:`str`
///     The time unit for the date. Must be one of ``'ms'`` or ``'days'``.
///
/// nullable : :class:`bool`
///     When :obj:`True`, :obj:`None` is a permissible value.
///
/// Returns
/// -------
/// :class:`vortex.DType`
///
/// Examples
/// --------
///
/// A data type representing dates as days since the UNIX epoch.
///
/// ```python
/// >>> import vortex as vx
/// >>> vx.date("days")
/// ext("vortex.date", int(32, nullable=False), days)
/// ```
#[pyfunction(name = "date")]
#[pyo3(signature = (unit, *, nullable = false))]
pub(super) fn dtype_date<'py>(
    py: Python<'py>,
    unit: &str,
    nullable: bool,
) -> PyVortexResult<Bound<'py, PyDType>> {
    let time_unit = parse_time_unit(unit)?;
    let ext = Date::try_new(time_unit, nullable.into())?;
    Ok(PyDType::init(py, DType::Extension(ext.erased()))?)
}

/// Construct a time-of-day data type.
///
/// Parameters
/// ----------
/// unit : :class:`str`
///     The time unit for the time. Must be one of ``'s'``, ``'ms'``, ``'us'``, or ``'ns'``.
///
/// nullable : :class:`bool`
///     When :obj:`True`, :obj:`None` is a permissible value.
///
/// Returns
/// -------
/// :class:`vortex.DType`
///
/// Examples
/// --------
///
/// A data type representing time of day in microseconds.
///
/// ```python
/// >>> import vortex as vx
/// >>> vx.time("us")
/// ext("vortex.time", int(64, nullable=False), µs)
/// ```
#[pyfunction(name = "time")]
#[pyo3(signature = (unit, *, nullable = false))]
pub(super) fn dtype_time<'py>(
    py: Python<'py>,
    unit: &str,
    nullable: bool,
) -> PyVortexResult<Bound<'py, PyDType>> {
    let time_unit = parse_time_unit(unit)?;
    let ext = Time::try_new(time_unit, nullable.into())?;
    Ok(PyDType::init(py, DType::Extension(ext.erased()))?)
}

/// Construct a timestamp data type.
///
/// Parameters
/// ----------
/// unit : :class:`str`
///     The time unit for the timestamp. Must be one of ``'s'``, ``'ms'``, ``'us'``, or ``'ns'``.
///
/// tz : :class:`str` or :obj:`None`
///     An optional timezone string (e.g. ``"UTC"``).
///
/// nullable : :class:`bool`
///     When :obj:`True`, :obj:`None` is a permissible value.
///
/// Returns
/// -------
/// :class:`vortex.DType`
///
/// Examples
/// --------
///
/// A non-nullable timestamp in microseconds with no timezone.
///
/// ```python
/// >>> import vortex as vx
/// >>> vx.timestamp("us")
/// ext("vortex.timestamp", int(64, nullable=False), µs)
/// ```
///
/// A nullable timestamp in seconds with a UTC timezone.
///
/// ```python
/// >>> vx.timestamp("s", tz="UTC", nullable=True)
/// ext("vortex.timestamp", int(64, nullable=True), s, tz=UTC)
/// ```
#[pyfunction(name = "timestamp")]
#[pyo3(signature = (unit, *, tz = None, nullable = false))]
pub(super) fn dtype_timestamp<'py>(
    py: Python<'py>,
    unit: &str,
    tz: Option<&str>,
    nullable: bool,
) -> PyResult<Bound<'py, PyDType>> {
    let time_unit = parse_time_unit(unit)?;
    let ext = Timestamp::new_with_tz(time_unit, tz.map(Arc::from), nullable.into());
    PyDType::init(py, DType::Extension(ext.erased()))
}
