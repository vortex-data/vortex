use std::sync::Arc;

use arrow::array::RecordBatchReader;
use arrow::datatypes::SchemaRef;
use arrow::pyarrow::{IntoPyArrow, ToPyArrow};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyLong, PyString};
use vortex::arrow::infer_schema;
use vortex::dtype::field::Field;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::file::{
    read_initial_bytes, LayoutContext, LayoutDeserializer, Projection, RowFilter,
    VortexFileArrayStream, VortexReadBuilder, VortexRecordBatchReader,
};
use vortex::io::{ObjectStoreReadAt, TokioFile, VortexReadAt};
use vortex::sampling_compressor::ALL_ENCODINGS_CONTEXT;
use vortex::ArrayData;

use crate::expr::PyExpr;
use crate::object_store_urls::vortex_read_at_from_url;
use crate::{PyArray, TOKIO_RUNTIME};

pub async fn layout_stream_from_reader<T: VortexReadAt + Unpin>(
    reader: T,
    projection: Projection,
    row_filter: Option<RowFilter>,
    indices: Option<ArrayData>,
) -> VortexResult<VortexFileArrayStream<T>> {
    let mut builder = VortexReadBuilder::new(
        reader,
        LayoutDeserializer::new(
            ALL_ENCODINGS_CONTEXT.clone(),
            LayoutContext::default().into(),
        ),
    )
    .with_projection(projection);

    if let Some(row_filter) = row_filter {
        builder = builder.with_row_filter(row_filter);
    }

    if let Some(indices) = indices {
        builder = builder.with_indices(indices);
    }

    builder.build().await
}

pub async fn read_array_from_reader<T: VortexReadAt + Unpin + 'static>(
    reader: T,
    projection: Projection,
    row_filter: Option<RowFilter>,
    indices: Option<ArrayData>,
) -> VortexResult<ArrayData> {
    layout_stream_from_reader(reader, projection, row_filter, indices)
        .await?
        .read_all()
        .await
}

pub async fn read_dtype_from_reader<T: VortexReadAt + Unpin>(reader: T) -> VortexResult<DType> {
    let initial_read = read_initial_bytes(&reader, reader.size().await?).await?;
    initial_read.lazy_dtype()?.value().cloned()
}

fn projection_from_python(columns: Option<Vec<Bound<PyAny>>>) -> PyResult<Projection> {
    fn field_from_pyany(field: &Bound<PyAny>) -> PyResult<Field> {
        if field.clone().is_instance_of::<PyString>() {
            Ok(Field::Name(
                field.downcast::<PyString>()?.to_str()?.to_string(),
            ))
        } else if field.is_instance_of::<PyLong>() {
            Ok(Field::Index(field.extract()?))
        } else {
            Err(PyTypeError::new_err(format!(
                "projection: expected list of string, int, and None, but found: {}.",
                field,
            )))
        }
    }

    Ok(match columns {
        None => Projection::All,
        Some(columns) => Projection::Flat(
            columns
                .iter()
                .map(field_from_pyany)
                .collect::<PyResult<Vec<Field>>>()?,
        ),
    })
}

fn row_filter_from_python(row_filter: Option<&Bound<PyExpr>>) -> Option<RowFilter> {
    row_filter.map(|x| RowFilter::new(x.borrow().unwrap().clone()))
}

#[pyclass(name = "TokioFileDataset", module = "io")]
pub struct TokioFileDataset {
    file: TokioFile,
    schema: SchemaRef,
}

impl TokioFileDataset {
    pub async fn try_new(path: String) -> VortexResult<Self> {
        let file = TokioFile::open(path)?;
        let schema = Arc::new(infer_schema(&read_dtype_from_reader(file.clone()).await?)?);

        Ok(Self { file, schema })
    }

    async fn async_to_array(
        &self,
        columns: Option<Vec<Bound<'_, PyAny>>>,
        row_filter: Option<&Bound<'_, PyExpr>>,
        indices: Option<&PyArray>,
    ) -> PyResult<PyArray> {
        let inner = read_array_from_reader(
            self.file.clone(),
            projection_from_python(columns)?,
            row_filter_from_python(row_filter),
            indices.map(PyArray::unwrap).cloned(),
        )
        .await?;
        Ok(PyArray::new(inner))
    }

    async fn async_to_record_batch_reader(
        self_: PyRef<'_, Self>,
        columns: Option<Vec<Bound<'_, PyAny>>>,
        row_filter: Option<&Bound<'_, PyExpr>>,
        indices: Option<&PyArray>,
    ) -> PyResult<PyObject> {
        let layout_reader = layout_stream_from_reader(
            self_.file.clone(),
            projection_from_python(columns)?,
            row_filter_from_python(row_filter),
            indices.map(PyArray::unwrap).cloned(),
        )
        .await?;

        let record_batch_reader: Box<dyn RecordBatchReader + Send> = Box::new(
            VortexRecordBatchReader::try_new(layout_reader, &*TOKIO_RUNTIME)?,
        );
        record_batch_reader.into_pyarrow(self_.py())
    }
}

#[pymethods]
impl TokioFileDataset {
    fn schema(self_: PyRef<Self>) -> PyResult<PyObject> {
        self_.schema.clone().to_pyarrow(self_.py())
    }

    #[pyo3(signature = (*, columns = None, row_filter = None, indices = None))]
    pub fn to_array(
        &self,
        columns: Option<Vec<Bound<'_, PyAny>>>,
        row_filter: Option<&Bound<'_, PyExpr>>,
        indices: Option<&PyArray>,
    ) -> PyResult<PyArray> {
        TOKIO_RUNTIME.block_on(self.async_to_array(columns, row_filter, indices))
    }

    #[pyo3(signature = (*, columns = None, row_filter = None, indices = None))]
    pub fn to_record_batch_reader(
        self_: PyRef<Self>,
        columns: Option<Vec<Bound<'_, PyAny>>>,
        row_filter: Option<&Bound<'_, PyExpr>>,
        indices: Option<&PyArray>,
    ) -> PyResult<PyObject> {
        TOKIO_RUNTIME.block_on(Self::async_to_record_batch_reader(
            self_, columns, row_filter, indices,
        ))
    }
}

#[pyclass(name = "ObjectStoreUrlDataset", module = "io")]
pub struct ObjectStoreUrlDataset {
    url: String,
    schema: SchemaRef,
}

impl ObjectStoreUrlDataset {
    async fn reader(&self) -> VortexResult<ObjectStoreReadAt> {
        vortex_read_at_from_url(&self.url).await
    }

    pub async fn try_new(url: String) -> VortexResult<Self> {
        let reader = vortex_read_at_from_url(&url).await?;
        let schema = Arc::new(infer_schema(&read_dtype_from_reader(reader).await?)?);

        Ok(Self { url, schema })
    }

    async fn async_to_array(
        &self,
        columns: Option<Vec<Bound<'_, PyAny>>>,
        row_filter: Option<&Bound<'_, PyExpr>>,
        indices: Option<&PyArray>,
    ) -> PyResult<PyArray> {
        let inner = read_array_from_reader(
            self.reader().await?,
            projection_from_python(columns)?,
            row_filter_from_python(row_filter),
            indices.map(PyArray::unwrap).cloned(),
        )
        .await?;
        Ok(PyArray::new(inner))
    }

    async fn async_to_record_batch_reader(
        self_: PyRef<'_, Self>,
        columns: Option<Vec<Bound<'_, PyAny>>>,
        row_filter: Option<&Bound<'_, PyExpr>>,
        indices: Option<&PyArray>,
    ) -> PyResult<PyObject> {
        let layout_reader = layout_stream_from_reader(
            self_.reader().await?,
            projection_from_python(columns)?,
            row_filter_from_python(row_filter),
            indices.map(PyArray::unwrap).cloned(),
        )
        .await?;

        let record_batch_reader: Box<dyn RecordBatchReader + Send> = Box::new(
            VortexRecordBatchReader::try_new(layout_reader, &*TOKIO_RUNTIME)?,
        );
        record_batch_reader.into_pyarrow(self_.py())
    }
}

#[pymethods]
impl ObjectStoreUrlDataset {
    fn schema(self_: PyRef<Self>) -> PyResult<PyObject> {
        self_.schema.clone().to_pyarrow(self_.py())
    }

    #[pyo3(signature = (*, columns = None, row_filter = None, indices = None))]
    pub fn to_array(
        &self,
        columns: Option<Vec<Bound<'_, PyAny>>>,
        row_filter: Option<&Bound<'_, PyExpr>>,
        indices: Option<&PyArray>,
    ) -> PyResult<PyArray> {
        TOKIO_RUNTIME.block_on(self.async_to_array(columns, row_filter, indices))
    }

    #[pyo3(signature = (*, columns = None, row_filter = None, indices = None))]
    pub fn to_record_batch_reader(
        self_: PyRef<Self>,
        columns: Option<Vec<Bound<'_, PyAny>>>,
        row_filter: Option<&Bound<'_, PyExpr>>,
        indices: Option<&PyArray>,
    ) -> PyResult<PyObject> {
        TOKIO_RUNTIME.block_on(Self::async_to_record_batch_reader(
            self_, columns, row_filter, indices,
        ))
    }
}

#[pyfunction]
pub fn dataset_from_url(url: Bound<PyString>) -> PyResult<ObjectStoreUrlDataset> {
    Ok(TOKIO_RUNTIME.block_on(ObjectStoreUrlDataset::try_new(url.extract()?))?)
}

#[pyfunction]
pub fn dataset_from_path(path: Bound<PyString>) -> PyResult<TokioFileDataset> {
    Ok(TOKIO_RUNTIME.block_on(TokioFileDataset::try_new(path.extract()?))?)
}
