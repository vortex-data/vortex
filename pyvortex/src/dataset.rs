use std::path::Path;
use std::sync::Arc;

use arrow::array::RecordBatchReader;
use arrow::datatypes::SchemaRef;
use arrow::pyarrow::{IntoPyArrow, ToPyArrow};
use pyo3::prelude::*;
use pyo3::pyfunction;
use pyo3::types::PyString;
use tokio::fs::File;
use vortex::arrow::infer_schema;
use vortex_dtype::field::Field;
use vortex_error::VortexResult;
use vortex_serde::io::{ObjectStoreReadAt, VortexReadAt};
use vortex_serde::layouts::{Projection, RowFilter, VortexRecordBatchReader};

use crate::expr::PyExpr;
use crate::io::{layout_stream_from_reader, read_array_from_reader, read_dtype_from_reader};
use crate::{PyArray, TOKIO_RUNTIME};

#[pyclass(name = "Dataset", module = "io")]
/// An on-disk Vortex dataset for use with an Arrow-compatible query engine.
pub struct PyDataset {
    reader: Arc<dyn VortexReadAt>,
    schema: SchemaRef,
}

impl PyDataset {
    async fn try_new(reader: Arc<dyn VortexReadAt>) -> VortexResult<PyDataset> {
        let dtype = read_dtype_from_reader(&reader).await?;
        let schema = infer_schema(&dtype)?;

        Ok(PyDataset {
            reader,
            schema: Arc::new(schema),
        })
    }
}

#[pymethods]
impl PyDataset {
    pub fn schema(self_: PyRef<Self>) -> PyResult<PyObject> {
        self_.schema.to_pyarrow(self_.py())
    }

    #[pyo3(signature = (columns, batch_size, row_filter))]
    pub fn to_array(
        &self,
        columns: Option<Vec<String>>,
        batch_size: Option<usize>,
        row_filter: Option<&Bound<PyExpr>>,
    ) -> PyResult<PyArray> {
        let projection = match columns {
            None => Projection::All,
            Some(columns) => {
                Projection::Flat(columns.into_iter().map(Field::Name).collect::<Vec<_>>())
            }
        };
        let row_filter = row_filter.map(|x| RowFilter::new(x.borrow().unwrap().clone()));
        let inner = TOKIO_RUNTIME.block_on(read_array_from_reader(
            self.reader.clone(),
            projection,
            batch_size,
            row_filter,
        ))?;
        Ok(PyArray::new(inner))
    }

    #[pyo3(signature = (columns, batch_size, row_filter))]
    pub fn to_record_batch_reader(
        self_: PyRef<Self>,
        columns: Option<Vec<String>>,
        batch_size: Option<usize>,
        row_filter: Option<&Bound<PyExpr>>,
    ) -> PyResult<PyObject> {
        let projection = match columns {
            None => Projection::All,
            Some(columns) => {
                Projection::Flat(columns.into_iter().map(Field::Name).collect::<Vec<_>>())
            }
        };

        let row_filter = row_filter.map(|x| RowFilter::new(x.borrow().unwrap().clone()));
        let reader = self_.reader.clone();

        let layout_reader = TOKIO_RUNTIME.block_on(layout_stream_from_reader(
            reader, projection, batch_size, row_filter,
        ))?;

        let record_batch_reader: Box<dyn RecordBatchReader + Send> = Box::new(
            VortexRecordBatchReader::new(layout_reader, &*TOKIO_RUNTIME)?,
        );

        record_batch_reader.into_pyarrow(self_.py())
    }
}

#[pyfunction]
pub fn dataset_from_url(url: Bound<PyString>) -> PyResult<PyDataset> {
    async fn f(url: &str) -> PyResult<PyDataset> {
        let reader = ObjectStoreReadAt::try_new_from_url(url).await?;
        Ok(PyDataset::try_new(Arc::new(reader)).await?)
    }

    TOKIO_RUNTIME.block_on(f(&url.extract::<String>()?))
}

#[pyfunction]
pub fn dataset_from_path(path: Bound<PyString>) -> PyResult<PyDataset> {
    async fn f(path: &str) -> PyResult<PyDataset> {
        let reader = File::open(Path::new(&path)).await?;
        Ok(PyDataset::try_new(Arc::new(reader)).await?)
    }

    TOKIO_RUNTIME.block_on(f(&path.extract::<String>()?))
}
