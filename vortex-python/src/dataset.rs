// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::RecordBatchReader;
use arrow_schema::SchemaRef;
use itertools::Itertools;
use pyo3::exceptions::PyTypeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::iter::ArrayIteratorExt;
use vortex::dtype::FieldName;
use vortex::dtype::FieldNames;
use vortex::error::VortexResult;
use vortex::expr::Expression;
use vortex::expr::root;
use vortex::expr::select;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::io::runtime::BlockingRuntime;
use vortex::layout::scan::split_by::SplitBy;

use crate::RUNTIME;
use crate::arrays::PyArrayRef;
use crate::arrow::IntoPyArrow;
use crate::arrow::ToPyArrow;
use crate::error::PyVortexResult;
use crate::expr::PyExpr;
use crate::install_module;
use crate::object_store::resolve::ResolvedStore;
use crate::object_store::resolve::resolve_store;
use crate::session::session;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "dataset")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.dataset", &m)?;

    m.add_class::<PyVortexDataset>()?;

    m.add_function(wrap_pyfunction!(dataset_from_url, &m)?)?;

    Ok(())
}

pub fn read_array_from_reader(
    vortex_file: &VortexFile,
    projection: Expression,
    filter: Option<Expression>,
    indices: Option<ArrayRef>,
    row_range: Option<(u64, u64)>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let mut scan = vortex_file.scan()?.with_projection(projection);

    if let Some(filter) = filter {
        scan = scan.with_filter(filter);
    }

    if let Some(indices) = indices {
        let primitive = indices.execute::<PrimitiveArray>(ctx)?;
        let indices = primitive.into_buffer();
        scan = scan.with_row_indices(indices);
    }

    if let Some((l, r)) = row_range {
        scan = scan.with_row_range(l..r);
    }

    scan.into_array_iter(&*RUNTIME)?.read_all()
}

fn projection_from_python(columns: Option<Vec<Bound<PyAny>>>) -> PyResult<Expression> {
    fn field_from_pyany(field: &Bound<PyAny>) -> PyResult<FieldName> {
        if field.clone().is_instance_of::<PyString>() {
            Ok(FieldName::from(field.cast::<PyString>()?.to_str()?))
        } else {
            Err(PyTypeError::new_err(format!(
                "projection: expected list of strings or None, but found: {field}.",
            )))
        }
    }

    Ok(match columns {
        None => root(),
        Some(columns) => {
            let fields: Vec<_> = columns
                .iter()
                .map(field_from_pyany)
                .collect::<PyResult<_>>()?;
            select(FieldNames::from(fields), root())
        }
    })
}

fn filter_from_python(row_filter: Option<&Bound<PyExpr>>) -> Option<Expression> {
    row_filter.map(|x| x.borrow().inner().clone())
}

#[pyclass(name = "VortexDataset", module = "dataset")]
pub struct PyVortexDataset {
    vxf: VortexFile,
    schema: SchemaRef,
}

impl PyVortexDataset {
    pub fn try_new(vxf: VortexFile) -> VortexResult<Self> {
        let schema = Arc::new(vxf.dtype().to_arrow_schema()?);
        Ok(Self { vxf, schema })
    }

    pub async fn from_url(
        url: &str,
        store: Option<Arc<dyn object_store::ObjectStore>>,
    ) -> VortexResult<Self> {
        let session = session();
        let vxf = match resolve_store(url, store)? {
            ResolvedStore::ObjectStore(store, path) => {
                session
                    .open_options()
                    .open_object_store(&store, path.as_ref())
                    .await?
            }
            ResolvedStore::Path(path) => session.open_options().open_path(path).await?,
        };
        PyVortexDataset::try_new(vxf)
    }

    pub(crate) fn to_array_inner<'py>(
        &self,
        py: Python<'py>,
        columns: Option<Vec<Bound<'py, PyAny>>>,
        row_filter: Option<&Bound<'py, PyExpr>>,
        indices: Option<PyArrayRef>,
        row_range: Option<(u64, u64)>,
    ) -> PyVortexResult<PyArrayRef> {
        let vxf = self.vxf.clone();
        let projection = projection_from_python(columns)?;
        let filter = filter_from_python(row_filter);
        let indices = indices.map(|i| i.into_inner());

        let array = py.detach(move || {
            let session = session();
            let mut ctx = session.create_execution_ctx();
            read_array_from_reader(&vxf, projection, filter, indices, row_range, &mut ctx)
        })?;
        Ok(PyArrayRef::from(array))
    }
}

#[pymethods]
impl PyVortexDataset {
    fn schema(self_: PyRef<Self>) -> PyResult<Py<PyAny>> {
        Arc::clone(&self_.schema).to_pyarrow(self_.py())
    }

    #[pyo3(signature = (*, columns = None, row_filter = None, indices = None, row_range = None))]
    pub fn to_array<'py>(
        self_: PyRef<'py, Self>,
        columns: Option<Vec<Bound<'py, PyAny>>>,
        row_filter: Option<&Bound<'py, PyExpr>>,
        indices: Option<PyArrayRef>,
        row_range: Option<(u64, u64)>,
    ) -> PyVortexResult<PyArrayRef> {
        self_.to_array_inner(self_.py(), columns, row_filter, indices, row_range)
    }

    #[pyo3(signature = (*, columns = None, row_filter = None, split_by = None, row_range = None))]
    pub fn to_record_batch_reader(
        self_: PyRef<Self>,
        columns: Option<Vec<Bound<'_, PyAny>>>,
        row_filter: Option<&Bound<'_, PyExpr>>,
        split_by: Option<usize>,
        row_range: Option<(u64, u64)>,
    ) -> PyVortexResult<Py<PyAny>> {
        let vxf = self_.vxf.clone();
        let projection = projection_from_python(columns)?;
        let filter = filter_from_python(row_filter);

        let reader = self_.py().detach(move || {
            let mut scan = vxf
                .scan()?
                .with_projection(projection)
                .with_some_filter(filter)
                .with_split_by(split_by.map(SplitBy::RowCount).unwrap_or(SplitBy::Layout));
            if let Some((l, r)) = row_range {
                scan = scan.with_row_range(l..r);
            }

            let schema = Arc::new(scan.dtype()?.to_arrow_schema()?);
            let reader: Box<dyn RecordBatchReader + Send> =
                Box::new(scan.into_record_batch_reader(schema, &*RUNTIME)?);
            VortexResult::Ok(reader)
        })?;

        Ok(reader.into_pyarrow(self_.py())?)
    }

    /// The number of rows matching the filter.
    #[pyo3(signature = (*, row_filter = None, split_by = None, row_range = None))]
    pub fn count_rows(
        self_: PyRef<Self>,
        row_filter: Option<&Bound<'_, PyExpr>>,
        split_by: Option<usize>,
        row_range: Option<(u64, u64)>,
    ) -> PyVortexResult<usize> {
        if row_filter.is_none() {
            let row_count = match row_range {
                Some(range) => range.1 - range.0,
                None => self_.vxf.row_count(),
            };
            return row_count
                .try_into()
                .map_err(|e| PyValueError::new_err(e).into());
        }

        let vxf = self_.vxf.clone();
        let filter = filter_from_python(row_filter);
        let n_rows: usize = self_.py().detach(move || {
            let mut scan = vxf
                .scan()?
                .with_projection(select(FieldNames::empty(), root()))
                .with_some_filter(filter)
                .with_split_by(split_by.map(SplitBy::RowCount).unwrap_or(SplitBy::Layout));
            if let Some((l, r)) = row_range {
                scan = scan.with_row_range(l..r);
            }

            scan.into_array_iter(&*RUNTIME)?
                .map_ok(|array| array.len())
                .process_results(|iter| iter.sum())
        })?;

        Ok(n_rows)
    }

    /// The natural splits of this Dataset.
    #[pyo3(signature = (*))]
    pub fn splits(&self) -> PyVortexResult<Vec<(u64, u64)>> {
        Ok(self
            .vxf
            .splits()?
            .into_iter()
            .map(|x| (x.start, x.end))
            .collect())
    }
}

#[pyfunction]
#[pyo3(signature = (url, *, store = None))]
pub fn dataset_from_url(
    py: Python,
    url: &str,
    store: Option<Bound<PyAny>>,
) -> PyVortexResult<PyVortexDataset> {
    let store_arc = if let Some(store_obj) = store {
        let py_store: pyo3_object_store::PyObjectStore = store_obj.extract()?;
        Some(py_store.into_inner())
    } else {
        None
    };

    Ok(py.detach(move || RUNTIME.block_on(PyVortexDataset::from_url(url, store_arc)))?)
}
