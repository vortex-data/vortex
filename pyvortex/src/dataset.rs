use std::pin::Pin;
use std::sync::Arc;

use arrow::array::RecordBatchReader;
use arrow::datatypes::SchemaRef;
use arrow::pyarrow::{IntoPyArrow, ToPyArrow};
use futures::TryStreamExt;
use pyo3::exceptions::PyTypeError;
use pyo3::ffi::Py_uintptr_t;
use pyo3::prelude::*;
use pyo3::types::{PyString, PyTuple};
use vortex::array::ChunkedArray;
use vortex::arrow::infer_schema;
use vortex::buffer::Buffer;
use vortex::dtype::{DType, FieldName};
use vortex::error::VortexResult;
use vortex::expr::{get_item, ident, pack, ExprRef, Identity};
use vortex::file::v2::{Scan, VortexOpenOptions};
use vortex::file::{read_initial_bytes, VortexRecordBatchReader};
use vortex::io::{ObjectStoreReadAt, TokioFile, VortexReadAt};
use vortex::stream::{ArrayStream, ArrayStreamAdapter};
use vortex::{ArrayData, IntoArrayData, IntoArrayVariant};

use crate::expr::PyExpr;
use crate::object_store_urls::vortex_read_at_from_url;
use crate::{PyArray, TOKIO_RUNTIME};

pub async fn layout_stream_from_reader<T: VortexReadAt + Unpin>(
    reader: T,
    projection: ExprRef,
    filter: Option<ExprRef>,
    indices: Option<ArrayData>,
) -> VortexResult<Pin<Box<dyn ArrayStream>>> {
    let vortex_file = VortexOpenOptions::new(Default::default())
        .open(reader)
        .await?;
    let scan = Scan::new(projection, filter);

    let array_stream = match indices {
        Some(take_buffer) => {
            let indices =
                Buffer::from_iter(take_buffer.into_primitive()?.as_slice::<u64>().to_vec());
            let s = vortex_file.take(indices, scan)?;
            Box::pin(ArrayStreamAdapter::new(vortex_file.dtype().clone(), s)) as _
        }
        None => {
            let s = vortex_file.scan(scan)?;
            Box::pin(ArrayStreamAdapter::new(vortex_file.dtype().clone(), s)) as _
        }
    };

    Ok(array_stream)
}

pub async fn read_array_from_reader<T: VortexReadAt + Unpin + 'static>(
    reader: T,
    projection: ExprRef,
    row_filter: Option<ExprRef>,
    indices: Option<ArrayData>,
) -> VortexResult<ArrayData> {
    let stream = layout_stream_from_reader(reader, projection, row_filter, indices).await?;
    let dtype = stream.dtype().clone();

    let data = stream.try_collect::<Vec<_>>().await?;

    match data.len() {
        1 => Ok(data[0].clone()),
        _ => Ok(ChunkedArray::try_new(data, dtype.clone())?.into_array()),
    }
}

pub async fn read_dtype_from_reader<T: VortexReadAt + Unpin>(reader: T) -> VortexResult<DType> {
    let initial_read = read_initial_bytes(&reader, reader.size().await?).await?;
    Ok(initial_read.dtype())
}

fn projection_from_python(columns: Option<Vec<Bound<PyAny>>>) -> PyResult<ExprRef> {
    fn field_from_pyany(field: &Bound<PyAny>) -> PyResult<FieldName> {
        if field.clone().is_instance_of::<PyString>() {
            Ok(FieldName::from(field.downcast::<PyString>()?.to_str()?))
        } else {
            Err(PyTypeError::new_err(format!(
                "projection: expected list of strings or None, but found: {}.",
                field,
            )))
        }
    }

    Ok(match columns {
        None => Identity::new_expr(),
        Some(columns) => {
            let fields = columns
                .iter()
                .map(field_from_pyany)
                .collect::<PyResult<Arc<[FieldName]>>>()?;

            pack(
                fields.clone(),
                fields
                    .iter()
                    .map(|f| get_item(f.clone(), ident()))
                    .collect(),
            )
        }
    })
}

fn filter_from_python(row_filter: Option<&Bound<PyExpr>>) -> Option<ExprRef> {
    row_filter.map(|x| x.borrow().unwrap().clone())
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
            filter_from_python(row_filter),
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
        let stream = layout_stream_from_reader(
            self_.file.clone(),
            projection_from_python(columns)?,
            filter_from_python(row_filter),
            indices.map(PyArray::unwrap).cloned(),
        )
        .await?;

        let record_batch_reader: Box<dyn RecordBatchReader + Send> =
            Box::new(VortexRecordBatchReader::try_new(stream, &*TOKIO_RUNTIME)?);

        record_batch_reader.into_pyarrow(self_.py())
    }
}

fn stream_to_arrow(stream: Pin<Box<dyn ArrayStream>>, py: Python) -> PyResult<PyObject> {
    let mut stream = arrow::ffi_stream::FFI_ArrowArrayStream::new(stream);

    let stream_ptr = (&mut stream) as *mut arrow::ffi_stream::FFI_ArrowArrayStream;
    let module = py.import_bound("pyarrow")?;
    let class = module.getattr("RecordBatchReader")?;
    let args = PyTuple::new_bound(py, [stream_ptr as Py_uintptr_t]);
    let reader = class.call_method1("_import_from_c", args)?;

    Ok(PyObject::from(reader))
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
            filter_from_python(row_filter),
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
        let stream = layout_stream_from_reader(
            self_.reader().await?,
            projection_from_python(columns)?,
            filter_from_python(row_filter),
            indices.map(PyArray::unwrap).cloned(),
        )
        .await?;

        let record_batch_reader: Box<dyn RecordBatchReader + Send> =
            Box::new(VortexRecordBatchReader::try_new(stream, &*TOKIO_RUNTIME)?);
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
