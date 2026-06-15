// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(crate) mod builtins;
pub(crate) mod compressed;
pub(crate) mod fastlanes;
pub(crate) mod from_arrow;
mod native;
pub(crate) mod py;
mod range_to_sequence;

use arrow_array::Array as ArrowArray;
use arrow_array::ArrayRef as ArrowArrayRef;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyIndexError;
use pyo3::exceptions::PyTypeError;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyList;
use pyo3::types::PyRange;
use pyo3::types::PyRangeMethods;
use pyo3_bytes::PyBytes;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::Chunked;
use vortex::array::arrays::bool::BoolArrayExt;
use vortex::array::arrays::chunked::ChunkedArrayExt;
use vortex::array::arrow::ArrowSessionExt;
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::match_each_integer_ptype;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::ipc::messages::EncoderMessage;
use vortex::ipc::messages::MessageEncoder;
use vortex::scalar_fn::fns::operators::Operator;

use crate::PyVortex;
use crate::arrays::native::PyNativeArray;
use crate::arrays::py::PyPythonArray;
use crate::arrays::py::PythonArray;
use crate::arrays::py::PythonVTable;
use crate::arrow::ToPyArrow;
use crate::dtype::PyDType;
use crate::error::PyVortexError;
use crate::error::PyVortexResult;
use crate::expr::PyExpr;
use crate::install_module;
use crate::python_repr::PythonRepr;
use crate::scalar::PyScalar;
use crate::serde::context::PyArrayContext;
use crate::session::session;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "arrays")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.arrays", &m)?;

    m.add_class::<PyArray>()?;
    m.add_class::<PyNativeArray>()?;
    m.add_class::<PyPythonArray>()?;

    // Canonical encodings
    m.add_class::<builtins::PyNullArray>()?;
    m.add_class::<builtins::PyBoolArray>()?;
    m.add_class::<builtins::PyPrimitiveArray>()?;
    m.add_class::<builtins::PyVarBinArray>()?;
    m.add_class::<builtins::PyVarBinViewArray>()?;
    m.add_class::<builtins::PyStructArray>()?;
    m.add_class::<builtins::PyListArray>()?;
    m.add_class::<builtins::PyFixedSizeListArray>()?;
    m.add_class::<builtins::PyExtensionArray>()?;

    // Utility encodings
    m.add_class::<builtins::PyConstantArray>()?;
    m.add_class::<builtins::PyChunkedArray>()?;
    m.add_class::<builtins::PyByteBoolArray>()?;

    // Compressed encodings
    m.add_class::<compressed::PyAlpArray>()?;
    m.add_class::<compressed::PyAlpRdArray>()?;
    m.add_class::<compressed::PyDateTimePartsArray>()?;
    m.add_class::<compressed::PyDictArray>()?;
    m.add_class::<compressed::PyFsstArray>()?;
    m.add_class::<compressed::PyRunEndArray>()?;
    m.add_class::<compressed::PySequenceArray>()?;
    m.add_class::<compressed::PySparseArray>()?;
    m.add_class::<compressed::PyZigZagArray>()?;

    // Fastlanes encodings
    m.add_class::<fastlanes::PyFastLanesBitPackedArray>()?;
    m.add_class::<fastlanes::PyFastLanesDeltaArray>()?;
    m.add_class::<fastlanes::PyFastLanesFoRArray>()?;

    Ok(())
}

/// A type adapter used to extract an ArrayRef from a Python object.
pub type PyArrayRef = PyVortex<ArrayRef>;

impl<'py> FromPyObject<'_, 'py> for PyArrayRef {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        // If it's already native, then we're done.
        if let Ok(native) = ob.cast::<PyNativeArray>() {
            return Ok(Self(native.get().inner().clone()));
        }

        // Otherwise, if it's a subclass of `PyArray`, then we can extract the inner array.
        PythonArray::extract(ob).map(|instance| Self(instance.into_array()))
    }
}

impl<'py> IntoPyObject<'py> for PyArrayRef {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyVortexError;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        // If the ArrayRef is a PyArrayInstance, extract the Python object.
        if let Some(pyarray) = self.0.as_opt::<PythonVTable>() {
            return pyarray.data().clone().into_pyobject(py);
        }

        // Otherwise, wrap the ArrayRef in a PyNativeArray.
        Ok(PyNativeArray::init(py, self.0)?.into_any())
    }
}

/// An array of zero or more *rows* each with the same set of *columns*.
///
/// Examples
/// --------
///
/// Arrays support all the standard comparison operations:
///
/// ```python
/// >>> import vortex as vx
/// >>> a = vx.array(['dog', None, 'cat', 'mouse', 'fish'])
/// >>> b = vx.array(['doug', 'jennifer', 'casper', 'mouse', 'faust'])
/// >>> (a < b).to_arrow_array()
/// <pyarrow.lib.BooleanArray object at ...>
/// [
///    true,
///    null,
///    false,
///    false,
///    false
/// ]
/// >>> (a <= b).to_arrow_array()
/// <pyarrow.lib.BooleanArray object at ...>
/// [
///    true,
///    null,
///    false,
///    true,
///    false
/// ]
/// >>> (a == b).to_arrow_array()
/// <pyarrow.lib.BooleanArray object at ...>
/// [
///    false,
///    null,
///    false,
///    true,
///    false
/// ]
/// >>> (a != b).to_arrow_array()
/// <pyarrow.lib.BooleanArray object at ...>
/// [
///    true,
///    null,
///    true,
///    false,
///    true
/// ]
/// >>> (a >= b).to_arrow_array()
/// <pyarrow.lib.BooleanArray object at ...>
/// [
///    false,
///    null,
///    true,
///    true,
///    true
/// ]
/// >>> (a > b).to_arrow_array()
/// <pyarrow.lib.BooleanArray object at ...>
/// [
///    false,
///    null,
///    true,
///    false,
///    true
/// ]
/// ```
#[pyclass(name = "Array", module = "vortex", sequence, subclass, frozen)]
pub struct PyArray;

#[pymethods]
impl PyArray {
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    #[expect(unused_variables)]
    fn new(args: &Bound<'_, PyAny>, kwargs: Option<&Bound<'_, PyAny>>) -> Self {
        Self
    }

    /// Convert a PyArrow object into a Vortex array.
    ///
    /// Parameters
    /// ----------
    /// obj: pyarrow.Array | pyarrow.ChunkedArray | pyarrow.Table
    ///     The array to convert.
    ///
    /// Returns
    /// -------
    /// :class:`~vortex.Array`
    #[staticmethod]
    fn from_arrow(obj: Bound<'_, PyAny>) -> PyVortexResult<PyArrayRef> {
        from_arrow::from_arrow(&obj.as_borrowed())
    }

    /// Convert a Python range into a Vortex array.
    ///
    /// Unless the array is empty, the encoding of the array is Sequence, which uses O(1) bytes to
    /// represent an array of any size.
    ///
    /// Parameters
    /// ----------
    /// range: range
    ///     The range to convert.
    ///
    /// Returns
    /// -------
    /// :class:`~vortex.Array`
    ///
    ///
    /// Examples
    /// --------
    ///
    /// ```python
    /// >>> array = vx.Array.from_range(range(0, 10))
    /// >>> array
    /// <vortex.SequenceArray object at ...>
    /// >>> array.to_arrow_array()
    /// <pyarrow.lib.Int64Array object at ...>
    /// [
    ///   0,
    ///   1,
    ///   2,
    ///   3,
    ///   4,
    ///   5,
    ///   6,
    ///   7,
    ///   8,
    ///   9
    /// ]
    /// ```
    #[staticmethod]
    #[pyo3(signature = (range, *, dtype = None))]
    fn from_range(
        range: Bound<PyAny>,
        dtype: Option<Bound<PyDType>>,
    ) -> PyVortexResult<PyArrayRef> {
        let range = range.cast::<PyRange>()?;
        let start = range.start()?;
        let stop = range.stop()?;
        let step = range.step()?;

        let (ptype, dtype) = if let Some(dtype) = dtype {
            let dtype = dtype.cast::<PyDType>()?.get().inner().clone();
            let DType::Primitive(ptype, ..) = &dtype else {
                return Err(PyValueError::new_err(
                    "Cannot construct non-numeric array from a range.",
                )
                .into());
            };
            (*ptype, dtype)
        } else {
            let ptype = if start > 0 && stop > 0 {
                PType::U64
            } else {
                PType::I64
            };
            let dtype = DType::Primitive(ptype, Nullability::NonNullable);
            (ptype, dtype)
        };

        let array = match_each_integer_ptype!(ptype, |T| {
            range_to_sequence::sequence_array_from_range::<T>(start, stop, step, dtype)
        })?;

        Ok(PyVortex(array))
    }

    /// Convert this array to a PyArrow array.
    ///
    /// .. seealso::
    ///     :meth:`.to_arrow_table`
    ///
    /// Returns
    /// -------
    /// :class:`pyarrow.Array`
    ///
    /// Examples
    /// --------
    ///
    /// Round-trip an Arrow array through a Vortex array:
    ///
    /// ```python
    /// >>> import vortex as vx
    /// >>> vx.array([1, 2, 3]).to_arrow_array()
    /// <pyarrow.lib.Int64Array object at ...>
    /// [
    ///   1,
    ///   2,
    ///   3
    /// ]
    /// ```
    fn to_arrow_array<'py>(self_: &'py Bound<'py, Self>) -> PyVortexResult<Bound<'py, PyAny>> {
        // NOTE(ngates): for struct arrays, we could also return a RecordBatchStreamReader.
        let array = PyArrayRef::extract(self_.as_any().as_borrowed())?.into_inner();
        let py = self_.py();

        if let Some(chunked_array) = array.as_opt::<Chunked>() {
            // We figure out a single Arrow Data Type to convert all chunks into, otherwise
            // the preferred type of each chunk may be different.
            let arrow_field = session()
                .arrow()
                .to_arrow_field("", chunked_array.dtype())?;

            let chunks = chunked_array
                .iter_chunks()
                .map(|chunk| -> PyVortexResult<_> {
                    Ok(session().arrow().execute_arrow(
                        chunk.clone(),
                        Some(&arrow_field),
                        &mut session().create_execution_ctx(),
                    )?)
                })
                .collect::<Result<Vec<ArrowArrayRef>, _>>()?;

            // NOTE(aduffy): pyarrow.chunked_array() constructor only takes DataType and not Field,
            //  which probably loses the extension information?
            let pa_data_type = arrow_field.data_type().clone().to_pyarrow(py)?;
            let chunks = chunks
                .iter()
                .map(|arrow_array| arrow_array.into_data().to_pyarrow(py))
                .collect::<Result<Vec<_>, _>>()?;

            let kwargs =
                PyDict::from_sequence(&PyList::new(py, vec![("type", pa_data_type)])?.into_any())?;

            // Combine into a chunked array
            Ok(PyModule::import(py, "pyarrow")?.call_method(
                "chunked_array",
                (PyList::new(py, chunks)?,),
                Some(&kwargs),
            )?)
        } else {
            Ok(session()
                .arrow()
                .execute_arrow(array, None, &mut session().create_execution_ctx())?
                .into_data()
                .to_pyarrow(py)?
                .into_bound(py))
        }
    }

    fn __len__(&self) -> PyResult<usize> {
        Err(PyTypeError::new_err("__len__ is not implemented for Array"))
    }

    fn __str__(&self) -> PyResult<String> {
        Err(PyTypeError::new_err("__str__ is not implemented for Array"))
    }

    /// Returns the encoding ID of this array.
    #[getter]
    fn id(slf: &Bound<Self>) -> PyResult<String> {
        Ok(PyArrayRef::extract(slf.as_any().as_borrowed())?
            .encoding_id()
            .to_string())
    }

    /// Returns the number of bytes used by this array.
    #[getter]
    fn nbytes(slf: &Bound<Self>) -> PyResult<u64> {
        Ok(PyArrayRef::extract(slf.as_any().as_borrowed())?.nbytes())
    }

    /// Returns the data type of this array.
    ///
    /// Returns
    /// -------
    /// :class:`vortex.DType`
    ///
    /// Examples
    /// --------
    ///
    /// By default, :func:`vortex.array` uses the largest available bit-width:
    ///
    /// ```python
    /// >>> import vortex as vx
    /// >>> vx.array([1, 2, 3]).dtype
    /// int(64, nullable=False)
    /// ```
    ///
    /// Including a :obj:`None` forces a nullable type:
    ///
    /// ```python
    /// >>> vx.array([1, None, 2, 3]).dtype
    /// int(64, nullable=True)
    /// ```
    ///
    /// A UTF-8 string array:
    ///
    /// ```python
    /// >>> vx.array(['hello, ', 'is', 'it', 'me?']).dtype
    /// utf8(nullable=False)
    /// ```
    #[getter]
    fn dtype<'py>(slf: &'py Bound<'py, Self>) -> PyResult<Bound<'py, PyDType>> {
        PyDType::init(
            slf.py(),
            PyArrayRef::extract(slf.as_any().as_borrowed())?
                .dtype()
                .clone(),
        )
    }

    /// Apply an expression on this array
    ///
    /// Examples
    /// --------
    ///
    /// Extract one column from a Vortex array:
    ///
    /// ```python
    /// >>> import vortex.expr as ve
    /// >>> import vortex as vx
    /// >>> array = vx.array([{"a": 0, "b": "hello"}, {"a": 1, "b": "goodbye"}])
    /// >>> expr = ve.column("a")
    /// >>> array = array.apply(expr)
    /// >>> array.to_arrow_array().to_pylist()
    /// [0, 1]
    /// ```
    ///
    /// See also
    /// --------
    /// vortex.open : Open an on-disk Vortex array for scanning with an expression.
    /// vortex.VortexFile : An on-disk Vortex array ready to scan with an expression.
    /// vortex.VortexFile.scan : Scan an on-disk Vortex array with an expression.
    pub fn apply(slf: Bound<Self>, expr: PyExpr) -> PyVortexResult<PyArrayRef> {
        let slf = PyArrayRef::extract(slf.as_any().as_borrowed())?.into_inner();

        let inner = slf.apply(&expr)?;

        Ok(PyArrayRef::from(inner))
    }

    /// Rust docs are *not* copied into Python for __lt__: https://github.com/PyO3/pyo3/issues/4326
    fn __lt__(slf: Bound<Self>, other: PyArrayRef) -> PyVortexResult<PyArrayRef> {
        let slf = PyArrayRef::extract(slf.as_any().as_borrowed())?.into_inner();
        let inner = slf.binary(other.into_inner(), Operator::Lt)?;
        Ok(PyArrayRef::from(inner))
    }

    /// Rust docs are *not* copied into Python for __le__: https://github.com/PyO3/pyo3/issues/4326
    fn __le__(slf: Bound<Self>, other: PyArrayRef) -> PyVortexResult<PyArrayRef> {
        let slf = PyArrayRef::extract(slf.as_any().as_borrowed())?.into_inner();
        let inner = slf.binary(other.into_inner(), Operator::Lte)?;
        Ok(PyArrayRef::from(inner))
    }

    /// Rust docs are *not* copied into Python for __eq__: https://github.com/PyO3/pyo3/issues/4326
    fn __eq__(slf: Bound<Self>, other: PyArrayRef) -> PyVortexResult<PyArrayRef> {
        let slf = PyArrayRef::extract(slf.as_any().as_borrowed())?.into_inner();
        let inner = slf.binary(other.into_inner(), Operator::Eq)?;
        Ok(PyArrayRef::from(inner))
    }

    /// Rust docs are *not* copied into Python for __ne__: https://github.com/PyO3/pyo3/issues/4326
    fn __ne__(slf: Bound<Self>, other: PyArrayRef) -> PyVortexResult<PyArrayRef> {
        let slf = PyArrayRef::extract(slf.as_any().as_borrowed())?.into_inner();
        let inner = slf.binary(other.into_inner(), Operator::NotEq)?;
        Ok(PyArrayRef::from(inner))
    }

    /// Rust docs are *not* copied into Python for __ge__: https://github.com/PyO3/pyo3/issues/4326
    fn __ge__(slf: Bound<Self>, other: PyArrayRef) -> PyVortexResult<PyArrayRef> {
        let slf = PyArrayRef::extract(slf.as_any().as_borrowed())?.into_inner();
        let inner = slf.binary(other.into_inner(), Operator::Gte)?;
        Ok(PyArrayRef::from(inner))
    }

    /// Rust docs are *not* copied into Python for __gt__: https://github.com/PyO3/pyo3/issues/4326
    fn __gt__(slf: Bound<Self>, other: PyArrayRef) -> PyVortexResult<PyArrayRef> {
        let slf = PyArrayRef::extract(slf.as_any().as_borrowed())?.into_inner();
        let inner = slf.binary(other.into_inner(), Operator::Gt)?;
        Ok(PyArrayRef::from(inner))
    }

    /// Filter an Array by another Boolean array.
    ///
    /// Parameters
    /// ----------
    /// filter : :class:`~vortex.Array`
    ///     Keep all the rows in ``self`` for which the correspondingly indexed row in `filter` is True.
    ///
    /// Returns
    /// -------
    /// :class:`~vortex.Array`
    ///
    /// Examples
    /// --------
    ///
    /// Keep only the single digit positive integers.
    ///
    /// ```python
    /// >>> import vortex as vx
    /// >>> a = vx.array([0, 42, 1_000, -23, 10, 9, 5])
    /// >>> filter = vx.array([True, False, False, False, False, True, True])
    /// >>> a.filter(filter).to_arrow_array()
    /// <pyarrow.lib.Int64Array object at ...>
    /// [
    ///   0,
    ///   9,
    ///   5
    /// ]
    /// ```
    fn filter(slf: Bound<Self>, mask: PyArrayRef) -> PyVortexResult<PyArrayRef> {
        // PyArray/PyArrayRef do not currently carry a VortexSession; threading one
        // through would change the FromPyObject contract. Use the crate session
        // until the wrappers are refactored.
        let mut ctx = session().create_execution_ctx();
        let slf = PyArrayRef::extract(slf.as_any().as_borrowed())?.into_inner();
        let mask_bool = (&*mask as &ArrayRef)
            .clone()
            .execute::<BoolArray>(&mut ctx)?;
        let mask = mask_bool.to_mask_fill_null_false(&mut ctx);
        let canonical = slf.filter(mask)?.execute::<Canonical>(&mut ctx)?;
        let inner = canonical.into_array();
        Ok(PyArrayRef::from(inner))
    }

    /// Retrieve a row by its index.
    ///
    /// Parameters
    /// ----------
    /// index : :class:`int`
    ///     The index of interest. Must be greater than or equal to zero and less than the length of
    ///     this array.
    ///
    /// Returns
    /// -------
    /// :class:`vortex.Scalar`
    ///
    /// Examples
    /// --------
    ///
    /// Retrieve the last element from an array of integers:
    ///
    /// ```python
    /// >>> import vortex as vx
    /// >>> vx.array([10, 42, 999, 1992]).scalar_at(3).as_py()
    /// 1992
    /// ```
    ///
    /// Retrieve the third element from an array of strings:
    ///
    /// ```python
    /// >>> array = vx.array(["hello", "goodbye", "it", "is"])
    /// >>> array.scalar_at(2).as_py()
    /// 'it'
    /// ```
    ///
    /// Retrieve an element from an array of structures:
    ///
    /// ```python
    /// >>> array = vx.array([
    /// ...     {'name': 'Joseph', 'age': 25},
    /// ...     {'name': 'Narendra', 'age': 31},
    /// ...     {'name': 'Angela', 'age': 33},
    /// ...     None,
    /// ...     {'name': 'Mikhail', 'age': 57},
    /// ... ])
    /// >>> array.scalar_at(2).as_py()
    /// {'age': 33, 'name': 'Angela'}
    /// ```
    ///
    /// Retrieve a missing element from an array of structures:
    ///
    /// ```python
    /// >>> array.scalar_at(3).as_py() is None
    /// True
    /// ```
    ///
    /// Out of bounds accesses are prohibited:
    ///
    /// ```python
    /// >>> vx.array([10, 42, 999, 1992]).scalar_at(10)
    /// Traceback (most recent call last):
    /// ...
    /// IndexError: Index 10 out of bounds from 0 to 4
    /// ```
    ///
    /// Unlike Python, negative indices are not supported:
    ///
    /// ```python
    /// >>> vx.array([10, 42, 999, 1992]).scalar_at(-2)
    /// Traceback (most recent call last):
    /// ...
    /// OverflowError: can't convert negative int to unsigned
    /// ```
    // TODO(ngates): return a vortex.Scalar
    fn scalar_at(slf: Bound<Self>, index: usize) -> PyVortexResult<Bound<PyScalar>> {
        // PyArray/PyArrayRef do not currently carry a VortexSession; threading one
        // through would change the FromPyObject contract. Use the crate session
        // until the wrappers are refactored.
        let mut ctx = session().create_execution_ctx();
        let py = slf.py();
        let slf = PyArrayRef::extract(slf.as_any().as_borrowed())?.into_inner();
        if index >= slf.len() {
            return Err(PyIndexError::new_err(format!(
                "Index {index} out of bounds from 0 to {}",
                slf.len()
            ))
            .into());
        }
        Ok(PyScalar::init(py, slf.execute_scalar(index, &mut ctx)?)?)
    }

    /// Filter, permute, and/or repeat elements by their index.
    ///
    /// Parameters
    /// ----------
    /// indices : :class:`~vortex.Array`
    ///     An array of indices to keep.
    ///
    /// Returns
    /// -------
    /// :class:`~vortex.Array`
    ///
    /// Examples
    /// --------
    ///
    /// Keep only the first and third elements:
    ///
    /// ```python
    /// >>> import vortex as vx
    /// >>> a = vx.array(['a', 'b', 'c', 'd'])
    /// >>> indices = vx.array([0, 2])
    /// >>> a.take(indices).to_arrow_array()
    /// <pyarrow.lib.StringViewArray object at ...>
    /// [
    ///   "a",
    ///   "c"
    /// ]
    /// ```
    ///
    /// Permute and repeat the first and second elements:
    ///
    /// ```python
    /// >>> a = vx.array(['a', 'b', 'c', 'd'])
    /// >>> indices = vx.array([0, 1, 1, 0])
    /// >>> a.take(indices).to_arrow_array()
    /// <pyarrow.lib.StringViewArray object at ...>
    /// [
    ///   "a",
    ///   "b",
    ///   "b",
    ///   "a"
    /// ]
    /// ```
    fn take(slf: Bound<Self>, indices: PyArrayRef) -> PyVortexResult<PyArrayRef> {
        let slf = PyArrayRef::extract(slf.as_any().as_borrowed())?.into_inner();

        if !indices.dtype().is_int() {
            return Err(PyValueError::new_err(format!(
                "indices: expected int or uint arra sy, but found: {}",
                indices.dtype().python_repr()
            ))
            .into());
        }

        let inner = slf.take(indices.clone())?;

        Ok(PyArrayRef::from(inner))
    }

    #[pyo3(signature = (start, end))]
    fn slice(slf: Bound<Self>, start: usize, end: usize) -> PyVortexResult<PyArrayRef> {
        let slf = PyArrayRef::extract(slf.as_any().as_borrowed())?.into_inner();
        let inner = slf.slice(start..end)?;
        Ok(PyArrayRef::from(inner))
    }

    /// Internal technical details about the encoding of this Array.
    ///
    /// Warnings
    /// --------
    /// The format of the returned string may change without notice.
    ///
    /// Returns
    /// -------
    /// :class:`.str`
    ///
    /// Examples
    /// --------
    ///
    /// Uncompressed arrays have straightforward encodings:
    ///
    /// ```python
    /// >>> import vortex as vx
    /// >>> arr = vx.array([1, 2, None, 3])
    /// >>> print(arr.display_tree()) # doctest: +ELLIPSIS
    /// root: vortex.primitive(i64?, len=4) nbytes=33 B (100.00%)
    ///   metadata: ptype: i64
    ///   buffer: values host 32 B (align=8) (96.97%)
    ///   validity: vortex.bool(bool, len=4) nbytes=1 B (3.03%)...
    ///     metadata: offset: 0
    ///     buffer: bits host 1 B (align=1) (100.00%)
    /// <BLANKLINE>
    /// ```
    ///
    /// Compressed arrays often have more complex, deeply nested encoding trees.
    fn display_tree(slf: &Bound<Self>) -> PyResult<String> {
        Ok(PyArrayRef::extract(slf.as_any().as_borrowed())?
            .display_tree()
            .to_string())
    }

    fn serialize(slf: &Bound<Self>, ctx: &PyArrayContext) -> PyVortexResult<Vec<Vec<u8>>> {
        // FIXME(ngates): do not copy to vec, use buffer protocol
        let array = PyArrayRef::extract(slf.as_any().as_borrowed())?;
        Ok(array
            .serialize(
                ctx,
                &vortex::session::VortexSession::empty(),
                &Default::default(),
            )?
            .into_iter()
            .map(|buffer| buffer.to_vec())
            .collect())
    }

    /// Support for Python's pickle protocol.
    ///
    /// This method serializes the array using Vortex IPC format and returns
    /// the data needed for pickle to reconstruct the array.
    fn __reduce__<'py>(
        slf: &'py Bound<'py, Self>,
    ) -> PyVortexResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let py = slf.py();
        let array = PyArrayRef::extract(slf.as_any().as_borrowed())?.into_inner();

        let mut encoder = MessageEncoder::new(vortex::session::VortexSession::empty());
        let buffers = encoder.encode(EncoderMessage::Array(&array))?;

        // Return buffers as a list instead of concatenating
        let array_buffers: Vec<Vec<u8>> = buffers.iter().map(|b| b.to_vec()).collect();

        let dtype_buffers = encoder.encode(EncoderMessage::DType(array.dtype()))?;
        let dtype_buffers: Vec<Vec<u8>> = dtype_buffers.iter().map(|b| b.to_vec()).collect();

        let vortex_module = PyModule::import(py, "vortex")?;
        let unpickle_fn = vortex_module.getattr(intern!(py, "_unpickle_array"))?;

        let args = (array_buffers, dtype_buffers).into_pyobject(py)?;
        Ok((unpickle_fn, args.into_any()))
    }

    /// Support for Python's pickle protocol for protocol >= 5
    ///
    /// uses PickleBuffer for out-of-band buffer transfer,
    /// which potentially avoids copying large data buffers.
    fn __reduce_ex__<'py>(
        slf: &'py Bound<'py, Self>,
        protocol: i32,
    ) -> PyVortexResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let py = slf.py();

        if protocol < 5 {
            return Self::__reduce__(slf);
        }

        let array = PyArrayRef::extract(slf.as_any().as_borrowed())?.into_inner();

        let mut encoder = MessageEncoder::new(vortex::session::VortexSession::empty());
        let array_buffers = encoder.encode(EncoderMessage::Array(&array))?;
        let dtype_buffers = encoder.encode(EncoderMessage::DType(array.dtype()))?;

        let pickle_module = PyModule::import(py, "pickle")?;
        let pickle_buffer_class = pickle_module.getattr(intern!(py, "PickleBuffer"))?;

        let mut pickle_buffers = Vec::new();
        for buf in array_buffers.into_iter() {
            // PyBytes wraps bytes::Bytes and implements the buffer protocol
            // This allows PickleBuffer to reference the data without copying
            let py_bytes = PyBytes::new(buf).into_py_any(py)?;
            let pickle_buffer = pickle_buffer_class.call1((py_bytes,))?;
            pickle_buffers.push(pickle_buffer);
        }

        let mut dtype_pickle_buffers = Vec::new();
        for buf in dtype_buffers.into_iter() {
            let py_bytes = PyBytes::new(buf).into_py_any(py)?;
            let pickle_buffer = pickle_buffer_class.call1((py_bytes,))?;
            dtype_pickle_buffers.push(pickle_buffer);
        }

        let vortex_module = PyModule::import(py, "vortex")?;
        let unpickle_fn = vortex_module.getattr(intern!(py, "_unpickle_array"))?;

        let args = (pickle_buffers, dtype_pickle_buffers).into_pyobject(py)?;
        Ok((unpickle_fn, args.into_any()))
    }
}
