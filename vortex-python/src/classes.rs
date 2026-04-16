// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Caching often accesses classes that are accessed across the C ABI

use pyo3::Bound;
use pyo3::Py;
use pyo3::PyResult;
use pyo3::Python;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyType;

/// Returns the pyarrow.DataType class
pub fn data_type_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    TYPE.import(py, "pyarrow", "DataType")
}

/// Returns the pyarrow.Field class
pub fn field_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    TYPE.import(py, "pyarrow", "Field")
}

/// Returns the pyarrow.Schema class
pub fn schema_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    TYPE.import(py, "pyarrow", "Schema")
}

/// Returns the pyarrow.Array class
pub fn array_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    TYPE.import(py, "pyarrow", "Array")
}

/// Returns the pyarrow.ChunkedArray class
pub fn chunked_array_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    TYPE.import(py, "pyarrow", "ChunkedArray")
}

/// Returns the pyarrow.RecordBatchReader class
pub fn record_batch_reader_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    TYPE.import(py, "pyarrow", "RecordBatchReader")
}

/// Returns the pyarrow.Table class
pub fn table_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    TYPE.import(py, "pyarrow", "Table")
}

/// Returns the pyarrow.Decimal class
pub fn decimal_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    TYPE.import(py, "decimal", "Decimal")
}
