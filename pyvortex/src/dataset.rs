use std::sync::Arc;

use arrow::array::RecordBatchReader;
use arrow::datatypes::SchemaRef;
use arrow::pyarrow::{IntoPyArrow, ToPyArrow};
use futures::TryStreamExt;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyString;
use vortex::arrays::ChunkedArray;
use vortex::dtype::FieldName;
use vortex::error::VortexResult;
use vortex::expr::{ExprRef, Select, ident};
use vortex::file::{FileType, GenericVortexFile, VortexFile, VortexOpenOptions};
use vortex::io::{ObjectStoreReadAt, TokioFile};
use vortex::stream::ArrayStream;
use vortex::{Array, ArrayRef, ToCanonical};

use crate::arrays::PyArrayRef;
use crate::expr::PyExpr;
use crate::object_store_urls::vortex_read_at_from_url;
use crate::record_batch_reader::VortexRecordBatchReader;
use crate::{TOKIO_RUNTIME, install_module};

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "dataset")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.dataset", &m)?;

    m.add_function(wrap_pyfunction!(dataset_from_url, &m)?)?;
    m.add_function(wrap_pyfunction!(dataset_from_path, &m)?)?;

    Ok(())
}

pub async fn read_array_from_reader<F: FileType>(
    vortex_file: &VortexFile<F>,
    projection: ExprRef,
    filter: Option<ExprRef>,
    indices: Option<ArrayRef>,
) -> VortexResult<ArrayRef> {
    let mut scan = vortex_file.scan().with_projection(projection);

    if let Some(filter) = filter {
        scan = scan.with_filter(filter);
    }

    if let Some(indices) = indices {
        let indices = indices.to_primitive()?.into_buffer();
        scan = scan.with_row_indices(indices);
    }

    let stream = scan.into_array_stream()?;
    let dtype = stream.dtype().clone();

    let all_arrays = stream.try_collect::<Vec<_>>().await?;

    match all_arrays.len() {
        1 => Ok(all_arrays[0].clone()),
        _ => Ok(ChunkedArray::try_new(all_arrays, dtype.clone())?.into_array()),
    }
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
        None => ident(),
        Some(columns) => {
            let fields = columns
                .iter()
                .map(field_from_pyany)
                .collect::<PyResult<Arc<[FieldName]>>>()?;

            Select::include_expr(fields, ident())
        }
    })
}

fn filter_from_python(row_filter: Option<&Bound<PyExpr>>) -> Option<ExprRef> {
    row_filter.map(|x| x.borrow().unwrap().clone())
}

#[pyclass(name = "TokioFileDataset", module = "io")]
pub struct TokioFileDataset {
    vxf: VortexFile<GenericVortexFile<TokioFile>>,
    schema: SchemaRef,
}

impl TokioFileDataset {
    pub async fn try_new(path: String) -> VortexResult<Self> {
        let file = TokioFile::open(path)?;
        let vxf = VortexOpenOptions::file(file).open().await?;
        let schema = Arc::new(vxf.dtype().to_arrow_schema()?);

        Ok(Self { vxf, schema })
    }

    async fn async_to_array(
        &self,
        columns: Option<Vec<Bound<'_, PyAny>>>,
        row_filter: Option<&Bound<'_, PyExpr>>,
        indices: Option<PyArrayRef>,
    ) -> PyResult<ArrayRef> {
        Ok(read_array_from_reader(
            &self.vxf,
            projection_from_python(columns)?,
            filter_from_python(row_filter),
            indices.map(|i| i.into_inner()),
        )
        .await?)
    }

    async fn async_to_record_batch_reader(
        self_: PyRef<'_, Self>,
        columns: Option<Vec<Bound<'_, PyAny>>>,
        row_filter: Option<&Bound<'_, PyExpr>>,
        indices: Option<PyArrayRef>,
    ) -> PyResult<PyObject> {
        let mut scan = self_
            .vxf
            .scan()
            .with_projection(projection_from_python(columns)?)
            .with_some_filter(filter_from_python(row_filter));

        if let Some(indices) = indices.map(|i| i.inner().clone()) {
            let indices = indices.to_primitive()?.into_buffer();
            scan = scan.with_row_indices(indices);
        }

        let stream = scan.into_array_stream()?;

        let record_batch_reader: Box<dyn RecordBatchReader + Send> =
            Box::new(VortexRecordBatchReader::try_new(stream, &*TOKIO_RUNTIME)?);

        record_batch_reader.into_pyarrow(self_.py())
    }
}

#[pymethods]
impl TokioFileDataset {
    fn schema(self_: PyRef<Self>) -> PyResult<PyObject> {
        self_.schema.clone().to_pyarrow(self_.py())
    }

    #[pyo3(signature = (*, columns = None, row_filter = None, indices = None))]
    pub fn to_array<'py>(
        &self,
        columns: Option<Vec<Bound<'py, PyAny>>>,
        row_filter: Option<&Bound<'py, PyExpr>>,
        indices: Option<PyArrayRef>,
    ) -> PyResult<PyArrayRef> {
        Ok(PyArrayRef::from(TOKIO_RUNTIME.block_on(
            self.async_to_array(columns, row_filter, indices),
        )?))
    }

    #[pyo3(signature = (*, columns = None, row_filter = None, indices = None))]
    pub fn to_record_batch_reader(
        self_: PyRef<Self>,
        columns: Option<Vec<Bound<'_, PyAny>>>,
        row_filter: Option<&Bound<'_, PyExpr>>,
        indices: Option<PyArrayRef>,
    ) -> PyResult<PyObject> {
        TOKIO_RUNTIME.block_on(Self::async_to_record_batch_reader(
            self_, columns, row_filter, indices,
        ))
    }
}

#[pyclass(name = "ObjectStoreUrlDataset", module = "io")]
pub struct ObjectStoreUrlDataset {
    vxf: VortexFile<GenericVortexFile<ObjectStoreReadAt>>,
    schema: SchemaRef,
}

impl ObjectStoreUrlDataset {
    pub async fn try_new(url: String) -> VortexResult<Self> {
        let reader = vortex_read_at_from_url(&url).await?;

        let vxf = VortexOpenOptions::file(reader).open().await?;
        let schema = Arc::new(vxf.dtype().to_arrow_schema()?);

        Ok(Self { vxf, schema })
    }

    async fn async_to_array(
        &self,
        columns: Option<Vec<Bound<'_, PyAny>>>,
        row_filter: Option<&Bound<'_, PyExpr>>,
        indices: Option<PyArrayRef>,
    ) -> PyResult<ArrayRef> {
        Ok(read_array_from_reader(
            &self.vxf,
            projection_from_python(columns)?,
            filter_from_python(row_filter),
            indices.map(|i| i.into_inner()),
        )
        .await?)
    }

    async fn async_to_record_batch_reader(
        self_: PyRef<'_, Self>,
        columns: Option<Vec<Bound<'_, PyAny>>>,
        filter: Option<&Bound<'_, PyExpr>>,
        indices: Option<PyArrayRef>,
    ) -> PyResult<PyObject> {
        let mut scan = self_
            .vxf
            .scan()
            .with_projection(projection_from_python(columns)?)
            .with_some_filter(filter_from_python(filter));

        if let Some(indices) = indices.map(|i| i.inner().clone()) {
            let indices = indices.to_primitive()?.into_buffer();
            scan = scan.with_row_indices(indices);
        }

        let stream = scan.into_array_stream()?;

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
    pub fn to_array<'py>(
        &self,
        columns: Option<Vec<Bound<'py, PyAny>>>,
        row_filter: Option<&Bound<'py, PyExpr>>,
        indices: Option<PyArrayRef>,
    ) -> PyResult<PyArrayRef> {
        Ok(PyArrayRef::from(TOKIO_RUNTIME.block_on(
            self.async_to_array(columns, row_filter, indices),
        )?))
    }

    #[pyo3(signature = (*, columns = None, row_filter = None, indices = None))]
    pub fn to_record_batch_reader(
        self_: PyRef<Self>,
        columns: Option<Vec<Bound<'_, PyAny>>>,
        row_filter: Option<&Bound<'_, PyExpr>>,
        indices: Option<PyArrayRef>,
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
