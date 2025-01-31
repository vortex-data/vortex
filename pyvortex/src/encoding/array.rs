use arrow::array::{Array as ArrowArray, ArrayRef};
use arrow::pyarrow::ToPyArrow;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyInt, PyList};
use vortex::array::ChunkedArray;
use vortex::arrow::{infer_data_type, IntoArrowArray};
use vortex::compute::{compare, fill_forward, scalar_at, slice, take, Operator};
use vortex::mask::Mask;
use vortex::Array;

use crate::dtype::PyDType;
use crate::python_repr::PythonRepr;
use crate::scalar::scalar_into_py;

#[pyclass(name = "Array", module = "vortex", sequence, subclass)]

pub struct PyArray {
    inner: Array,
}

impl PyArray {
    pub fn new(inner: Array) -> PyArray {
        PyArray { inner }
    }

    pub fn unwrap(&self) -> &Array {
        &self.inner
    }
}

#[pymethods]
impl PyArray {
    /// Convert this array to an Arrow array.
    fn to_arrow_array(self_: PyRef<'_, Self>) -> PyResult<Bound<PyAny>> {
        // NOTE(ngates): for struct arrays, we could also return a RecordBatchStreamReader.
        let py = self_.py();
        let vortex = &self_.inner;

        if let Ok(chunked_array) = ChunkedArray::try_from(vortex.clone()) {
            // We figure out a single Arrow Data Type to convert all chunks into, otherwise
            // the preferred type of each chunk may be different.
            let arrow_dtype = infer_data_type(chunked_array.dtype())?;

            let chunks: Vec<ArrayRef> = chunked_array
                .chunks()
                .map(|chunk| -> PyResult<ArrayRef> { Ok(chunk.into_arrow(&arrow_dtype)?) })
                .collect::<PyResult<Vec<ArrayRef>>>()?;
            if chunks.is_empty() {
                return Err(PyValueError::new_err("No chunks in array"));
            }
            let pa_data_type = chunks[0].data_type().clone().to_pyarrow(py)?;
            let chunks: PyResult<Vec<PyObject>> = chunks
                .iter()
                .map(|arrow_array| arrow_array.into_data().to_pyarrow(py))
                .collect();

            // Combine into a chunked array
            PyModule::import_bound(py, "pyarrow")?.call_method(
                "chunked_array",
                (PyList::new_bound(py, chunks?),),
                Some(&[("type", pa_data_type)].into_py_dict_bound(py)),
            )
        } else {
            Ok(vortex
                .clone()
                .into_arrow_preferred()?
                .into_data()
                .to_pyarrow(py)?
                .into_bound(py))
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    #[getter]
    fn encoding(&self) -> String {
        self.inner.encoding().to_string()
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.inner.nbytes()
    }

    /// The data type of this array.
    #[getter]
    fn dtype(self_: PyRef<Self>) -> PyResult<Py<PyDType>> {
        PyDType::wrap(self_.py(), self_.inner.dtype().clone())
    }

    // Rust docs are *not* copied into Python for __lt__: https://github.com/PyO3/pyo3/issues/4326
    fn __lt__(&self, other: &Bound<PyArray>) -> PyResult<PyArray> {
        let other = other.borrow();
        let inner = compare(&self.inner, &other.inner, Operator::Lt)?;
        Ok(PyArray { inner })
    }

    // Rust docs are *not* copied into Python for __le__: https://github.com/PyO3/pyo3/issues/4326
    fn __le__(&self, other: &Bound<PyArray>) -> PyResult<PyArray> {
        let other = other.borrow();
        let inner = compare(&self.inner, &other.inner, Operator::Lte)?;
        Ok(PyArray { inner })
    }

    // Rust docs are *not* copied into Python for __eq__: https://github.com/PyO3/pyo3/issues/4326
    fn __eq__(&self, other: &Bound<PyArray>) -> PyResult<PyArray> {
        let other = other.borrow();
        let inner = compare(&self.inner, &other.inner, Operator::Eq)?;
        Ok(PyArray { inner })
    }

    // Rust docs are *not* copied into Python for __ne__: https://github.com/PyO3/pyo3/issues/4326
    fn __ne__(&self, other: &Bound<PyArray>) -> PyResult<PyArray> {
        let other = other.borrow();
        let inner = compare(&self.inner, &other.inner, Operator::NotEq)?;
        Ok(PyArray { inner })
    }

    // Rust docs are *not* copied into Python for __ge__: https://github.com/PyO3/pyo3/issues/4326
    fn __ge__(&self, other: &Bound<PyArray>) -> PyResult<PyArray> {
        let other = other.borrow();
        let inner = compare(&self.inner, &other.inner, Operator::Gte)?;
        Ok(PyArray { inner })
    }

    // Rust docs are *not* copied into Python for __gt__: https://github.com/PyO3/pyo3/issues/4326
    fn __gt__(&self, other: &Bound<PyArray>) -> PyResult<PyArray> {
        let other = other.borrow();
        let inner = compare(&self.inner, &other.inner, Operator::Gt)?;
        Ok(PyArray { inner })
    }

    /// Filter an Array by another Boolean array.
    fn filter(&self, mask: &Bound<PyArray>) -> PyResult<PyArray> {
        let mask = mask.borrow();
        let inner = vortex::compute::filter(&self.inner, &Mask::try_from(mask.inner.clone())?)?;
        Ok(PyArray { inner })
    }

    /// Fill forward non-null values over runs of nulls.
    fn fill_forward(&self) -> PyResult<PyArray> {
        let inner = fill_forward(&self.inner)?;
        Ok(PyArray { inner })
    }

    /// Retrieve a row by its index.
    // TODO(ngates): return a vortex.Scalar
    fn scalar_at(&self, index: &Bound<PyInt>) -> PyResult<PyObject> {
        let scalar = scalar_at(&self.inner, index.extract()?)?;
        scalar_into_py(index.py(), scalar, false)
    }

    /// Filter, permute, and/or repeat elements by their index.
    fn take(&self, indices: &Bound<PyArray>) -> PyResult<PyArray> {
        let indices = &indices.borrow().inner;

        if !indices.dtype().is_int() {
            return Err(PyValueError::new_err(format!(
                "indices: expected int or uint array, but found: {}",
                indices.dtype().python_repr()
            )));
        }

        let inner = take(&self.inner, indices)?;
        Ok(PyArray { inner })
    }

    /// Keep only a contiguous subset of elements.
    #[pyo3(signature = (start, end))]
    fn slice(&self, start: usize, end: usize) -> PyResult<PyArray> {
        let inner = slice(&self.inner, start, end)?;
        Ok(PyArray::new(inner))
    }

    /// Internal technical details about the encoding of this Array.
    fn tree_display(&self) -> String {
        self.inner.tree_display().to_string()
    }
}
