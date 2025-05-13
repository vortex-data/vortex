pub(crate) mod builtins;
pub(crate) mod compressed;
pub(crate) mod fastlanes;
pub(crate) mod from_arrow;
mod native;
pub(crate) mod py;

use arrow::array::{Array as ArrowArray, ArrayRef as ArrowArrayRef};
use arrow::pyarrow::ToPyArrow;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use vortex::arrays::ChunkedVTable;
use vortex::arrow::IntoArrowArray;
use vortex::compute::{Operator, compare, take};
use vortex::error::VortexError;
use vortex::mask::Mask;
use vortex::nbytes::NBytes;
use vortex::{Array, ArrayExt, ArrayRef};

use crate::arrays::native::PyNativeArray;
use crate::arrays::py::{PyPythonArray, PythonArray};
use crate::dtype::PyDType;
use crate::python_repr::PythonRepr;
use crate::scalar::PyScalar;
use crate::serde::context::PyArrayContext;
use crate::{PyVortex, install_module};

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

impl<'py> FromPyObject<'py> for PyArrayRef {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        // If it's already native, then we're done.
        if let Ok(native) = ob.downcast::<PyNativeArray>() {
            return Ok(Self(native.get().inner().clone()));
        }

        // Otherwise, if it's a subclass of `PyArray`, then we can extract the inner array.
        PythonArray::extract_bound(ob).map(|instance| Self(instance.to_array()))
    }
}

impl<'py> IntoPyObject<'py> for PyArrayRef {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = VortexError;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        // If the ArrayRef is a PyArrayInstance, extract the Python object.
        if let Some(pyarray) = self.0.as_any().downcast_ref::<PythonArray>() {
            return pyarray.clone().into_pyobject(py);
        }

        // Otherwise, wrap the ArrayRef in a PyNativeArray.
        Ok(PyNativeArray::init(py, self.0.clone())?.into_any())
    }
}

/// An array of zero or more *rows* each with the same set of *columns*.
///
/// Examples
/// --------
///
/// Arrays support all the standard comparison operations:
///
///     >>> import vortex as vx
///     >>> a = vx.array(['dog', None, 'cat', 'mouse', 'fish'])
///     >>> b = vx.array(['doug', 'jennifer', 'casper', 'mouse', 'faust'])
///     >>> (a < b).to_arrow_array()
///     <pyarrow.lib.BooleanArray object at ...>
///     [
///        true,
///        null,
///        false,
///        false,
///        false
///     ]
///     >>> (a <= b).to_arrow_array()
///     <pyarrow.lib.BooleanArray object at ...>
///     [
///        true,
///        null,
///        false,
///        true,
///        false
///     ]
///     >>> (a == b).to_arrow_array()
///     <pyarrow.lib.BooleanArray object at ...>
///     [
///        false,
///        null,
///        false,
///        true,
///        false
///     ]
///     >>> (a != b).to_arrow_array()
///     <pyarrow.lib.BooleanArray object at ...>
///     [
///        true,
///        null,
///        true,
///        false,
///        true
///     ]
///     >>> (a >= b).to_arrow_array()
///     <pyarrow.lib.BooleanArray object at ...>
///     [
///        false,
///        null,
///        true,
///        true,
///        true
///     ]
///     >>> (a > b).to_arrow_array()
///     <pyarrow.lib.BooleanArray object at ...>
///     [
///        false,
///        null,
///        true,
///        false,
///        true
///     ]
#[pyclass(name = "Array", module = "vortex", sequence, subclass, frozen)]
pub struct PyArray;

#[pymethods]
impl PyArray {
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    #[allow(unused_variables)]
    fn new(args: &Bound<'_, PyAny>, kwargs: Option<&Bound<'_, PyAny>>) -> Self {
        Self
    }

    /// Convert a PyArrow object into a Vortex array.
    ///
    /// One of :class:`pyarrow.Array`, :class:`pyarrow.ChunkedArray`, or :class:`pyarrow.Table`.
    ///
    /// Returns
    /// -------
    /// :class:`~vortex.Array`
    #[staticmethod]
    fn from_arrow(obj: Bound<'_, PyAny>) -> PyResult<PyArrayRef> {
        from_arrow::from_arrow(&obj)
    }

    /// Convert this array to a PyArrow array.
    ///
    /// Convert this array to an Arrow array.
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
    ///     >>> import vortex as vx
    ///     >>> vx.array([1, 2, 3]).to_arrow_array()
    ///     <pyarrow.lib.Int64Array object at ...>
    ///     [
    ///       1,
    ///       2,
    ///       3
    ///     ]
    fn to_arrow_array<'py>(self_: &'py Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        // NOTE(ngates): for struct arrays, we could also return a RecordBatchStreamReader.
        let array = PyArrayRef::extract_bound(self_.as_any())?.into_inner();
        let py = self_.py();

        if let Some(chunked_array) = array.as_opt::<ChunkedVTable>() {
            // We figure out a single Arrow Data Type to convert all chunks into, otherwise
            // the preferred type of each chunk may be different.
            let arrow_dtype = chunked_array.dtype().to_arrow_dtype()?;

            let chunks = chunked_array
                .chunks()
                .iter()
                .map(|chunk| PyResult::Ok(chunk.clone().into_arrow(&arrow_dtype)?))
                .collect::<PyResult<Vec<ArrowArrayRef>>>()?;

            let pa_data_type = arrow_dtype.clone().to_pyarrow(py)?;
            let chunks = chunks
                .iter()
                .map(|arrow_array| arrow_array.into_data().to_pyarrow(py))
                .collect::<Result<Vec<_>, _>>()?;

            let kwargs =
                PyDict::from_sequence(&PyList::new(py, vec![("type", pa_data_type)])?.into_any())?;

            // Combine into a chunked array
            PyModule::import(py, "pyarrow")?.call_method(
                "chunked_array",
                (PyList::new(py, chunks)?,),
                Some(&kwargs),
            )
        } else {
            Ok(array
                .clone()
                .into_arrow_preferred()?
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
        Ok(PyArrayRef::extract_bound(slf.as_any())?
            .encoding_id()
            .to_string())
    }

    /// Returns the number of bytes used by this array.
    #[getter]
    fn nbytes(slf: &Bound<Self>) -> PyResult<usize> {
        Ok(PyArrayRef::extract_bound(slf.as_any())?.nbytes())
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
    ///     >>> import vortex as vx
    ///     >>> vx.array([1, 2, 3]).dtype
    ///     int(64, nullable=False)
    ///
    /// Including a :obj:`None` forces a nullable type:
    ///
    ///     >>> vx.array([1, None, 2, 3]).dtype
    ///     int(64, nullable=True)
    ///
    /// A UTF-8 string array:
    ///
    ///     >>> vx.array(['hello, ', 'is', 'it', 'me?']).dtype
    ///     utf8(nullable=False)
    #[getter]
    fn dtype<'py>(slf: &'py Bound<'py, Self>) -> PyResult<Bound<'py, PyDType>> {
        PyDType::init(
            slf.py(),
            PyArrayRef::extract_bound(slf.as_any())?.dtype().clone(),
        )
    }

    ///Rust docs are *not* copied into Python for __lt__: https://github.com/PyO3/pyo3/issues/4326
    fn __lt__(slf: Bound<Self>, other: PyArrayRef) -> PyResult<PyArrayRef> {
        let slf = PyArrayRef::extract_bound(slf.as_any())?.into_inner();
        let inner = compare(&slf, &*other, Operator::Lt)?;
        Ok(PyArrayRef::from(inner))
    }

    ///Rust docs are *not* copied into Python for __le__: https://github.com/PyO3/pyo3/issues/4326
    fn __le__(slf: Bound<Self>, other: PyArrayRef) -> PyResult<PyArrayRef> {
        let slf = PyArrayRef::extract_bound(slf.as_any())?.into_inner();
        let inner = compare(&*slf, &*other, Operator::Lte)?;
        Ok(PyArrayRef::from(inner))
    }

    ///Rust docs are *not* copied into Python for __eq__: https://github.com/PyO3/pyo3/issues/4326
    fn __eq__(slf: Bound<Self>, other: PyArrayRef) -> PyResult<PyArrayRef> {
        let slf = PyArrayRef::extract_bound(slf.as_any())?.into_inner();
        let inner = compare(&*slf, &*other, Operator::Eq)?;
        Ok(PyArrayRef::from(inner))
    }

    ///Rust docs are *not* copied into Python for __ne__: https://github.com/PyO3/pyo3/issues/4326
    fn __ne__(slf: Bound<Self>, other: PyArrayRef) -> PyResult<PyArrayRef> {
        let slf = PyArrayRef::extract_bound(slf.as_any())?.into_inner();
        let inner = compare(&*slf, &*other, Operator::NotEq)?;
        Ok(PyArrayRef::from(inner))
    }

    ///Rust docs are *not* copied into Python for __ge__: https://github.com/PyO3/pyo3/issues/4326
    fn __ge__(slf: Bound<Self>, other: PyArrayRef) -> PyResult<PyArrayRef> {
        let slf = PyArrayRef::extract_bound(slf.as_any())?.into_inner();
        let inner = compare(&*slf, &*other, Operator::Gte)?;
        Ok(PyArrayRef::from(inner))
    }

    ///Rust docs are *not* copied into Python for __gt__: https://github.com/PyO3/pyo3/issues/4326
    fn __gt__(slf: Bound<Self>, other: PyArrayRef) -> PyResult<PyArrayRef> {
        let slf = PyArrayRef::extract_bound(slf.as_any())?.into_inner();
        let inner = compare(&*slf, &*other, Operator::Gt)?;
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
    ///     >>> import vortex as vx
    ///     >>> a = vx.array([0, 42, 1_000, -23, 10, 9, 5])
    ///     >>> filter = vx.array([True, False, False, False, False, True, True])
    ///     >>> a.filter(filter).to_arrow_array()
    ///     <pyarrow.lib.Int64Array object at ...>
    ///     [
    ///       0,
    ///       9,
    ///       5
    ///     ]
    fn filter(slf: Bound<Self>, mask: PyArrayRef) -> PyResult<PyArrayRef> {
        let slf = PyArrayRef::extract_bound(slf.as_any())?.into_inner();
        let inner = vortex::compute::filter(&*slf, &Mask::try_from(&*mask as &dyn Array)?)?;
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
    ///     >>> import vortex as vx
    ///     >>> vx.array([10, 42, 999, 1992]).scalar_at(3).as_py()
    ///     1992
    ///
    /// Retrieve the third element from an array of strings:
    ///
    ///     >>> array = vx.array(["hello", "goodbye", "it", "is"])
    ///     >>> array.scalar_at(2).as_py()
    ///     'it'
    ///
    /// Retrieve an element from an array of structures:
    ///
    ///     >>> array = vx.array([
    ///     ...     {'name': 'Joseph', 'age': 25},
    ///     ...     {'name': 'Narendra', 'age': 31},
    ///     ...     {'name': 'Angela', 'age': 33},
    ///     ...     None,
    ///     ...     {'name': 'Mikhail', 'age': 57},
    ///     ... ])
    ///     >>> array.scalar_at(2).as_py()
    ///     {'age': 33, 'name': 'Angela'}
    ///
    /// Retrieve a missing element from an array of structures:
    ///
    ///     >>> array.scalar_at(3).as_py() is None
    ///     True
    ///
    /// Out of bounds accesses are prohibited:
    ///
    ///     >>> vx.array([10, 42, 999, 1992]).scalar_at(10)
    ///     Traceback (most recent call last):
    ///     ...
    ///     ValueError: index 10 out of bounds from 0 to 4
    ///     ...
    ///
    /// Unlike Python, negative indices are not supported:
    ///
    ///     >>> vx.array([10, 42, 999, 1992]).scalar_at(-2)
    ///     Traceback (most recent call last):
    ///     ...
    ///     OverflowError: can't convert negative int to unsigned
    // TODO(ngates): return a vortex.Scalar
    fn scalar_at(slf: Bound<Self>, index: usize) -> PyResult<Bound<PyScalar>> {
        let py = slf.py();
        let slf = PyArrayRef::extract_bound(slf.as_any())?.into_inner();
        PyScalar::init(py, slf.scalar_at(index)?)
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
    ///     >>> a = vx.array(['a', 'b', 'c', 'd'])
    ///     >>> indices = vx.array([0, 2])
    ///     >>> a.take(indices).to_arrow_array()
    ///     <pyarrow.lib.StringArray object at ...>
    ///     [
    ///       "a",
    ///       "c"
    ///     ]
    ///
    /// Permute and repeat the first and second elements:
    ///
    ///     >>> a = vx.array(['a', 'b', 'c', 'd'])
    ///     >>> indices = vx.array([0, 1, 1, 0])
    ///     >>> a.take(indices).to_arrow_array()
    ///     <pyarrow.lib.StringArray object at ...>
    ///     [
    ///       "a",
    ///       "b",
    ///       "b",
    ///       "a"
    ///     ]
    fn take(slf: Bound<Self>, indices: PyArrayRef) -> PyResult<PyArrayRef> {
        let slf = PyArrayRef::extract_bound(slf.as_any())?.into_inner();

        if !indices.dtype().is_int() {
            return Err(PyValueError::new_err(format!(
                "indices: expected int or uint array, but found: {}",
                indices.dtype().python_repr()
            )));
        }

        let inner = take(&slf, &*indices)?;

        Ok(PyArrayRef::from(inner))
    }

    /// Slice this array.
    ///
    /// Parameters
    /// ----------
    /// start : :class:`int`
    ///     The start index of the range to keep, inclusive.
    ///
    /// end : :class:`int`
    ///     The end index, exclusive.
    ///
    /// Returns
    /// -------
    /// :class:`~vortex.Array`
    ///
    /// Examples
    /// --------
    ///
    /// Keep only the second through third elements:
    ///
    ///     >>> import vortex as vx
    ///     >>> a = vx.array(['a', 'b', 'c', 'd'])
    ///     >>> a.slice(1, 3).to_arrow_array()
    ///     <pyarrow.lib.StringArray object at ...>
    ///     [
    ///       "b",
    ///       "c"
    ///     ]
    ///
    /// Keep none of the elements:
    ///
    ///     >>> a = vx.array(['a', 'b', 'c', 'd'])
    ///     >>> a.slice(3, 3).to_arrow_array()
    ///     <pyarrow.lib.StringViewArray object at ...>
    ///     []
    ///
    /// Unlike Python, it is an error to slice outside the bounds of the array:
    ///
    ///     >>> a = vx.array(['a', 'b', 'c', 'd'])
    ///     >>> a.slice(2, 10).to_arrow_array()
    ///     Traceback (most recent call last):
    ///     ...
    ///     ValueError: index 10 out of bounds from 0 to 4
    ///
    /// Or to slice with a negative value:
    ///
    ///     >>> a = vx.array(['a', 'b', 'c', 'd'])
    ///     >>> a.slice(-2, -1).to_arrow_array()
    ///     Traceback (most recent call last):
    ///     ...
    ///     OverflowError: can't convert negative int to unsigned
    #[pyo3(signature = (start, end))]
    fn slice(slf: Bound<Self>, start: usize, end: usize) -> PyResult<PyArrayRef> {
        let slf = PyArrayRef::extract_bound(slf.as_any())?.into_inner();
        let inner = slf.slice(start, end)?;
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
    ///     >>> import vortex as vx
    ///     >>> arr = vx.array([1, 2, None, 3])
    ///     >>> print(arr.tree_display())
    ///     root: vortex.primitive(i64?, len=4) nbytes=33 B (100.00%)
    ///       metadata: EmptyMetadata
    ///       buffer (align=8): 32 B (96.97%)
    ///       validity: vortex.bool(bool, len=4) nbytes=1 B (3.03%)
    ///         metadata: BoolMetadata { offset: 0 }
    ///         buffer (align=1): 1 B (100.00%)
    ///     <BLANKLINE>
    ///
    /// Compressed arrays often have more complex, deeply nested encoding trees.
    fn tree_display(slf: &Bound<Self>) -> PyResult<String> {
        Ok(PyArrayRef::extract_bound(slf.as_any())?
            .tree_display()
            .to_string())
    }

    fn serialize(slf: &Bound<Self>, ctx: &PyArrayContext) -> PyResult<Vec<Vec<u8>>> {
        // FIXME(ngates): do not copy to vec, use buffer protocol
        let array = PyArrayRef::extract_bound(slf.as_any())?;
        Ok(array
            .serialize(ctx, &Default::default())?
            .into_iter()
            .map(|buffer| buffer.to_vec())
            .collect())
    }
}
