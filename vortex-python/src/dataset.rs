// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::RecordBatchReader;
use arrow_schema::SchemaRef;
use itertools::Itertools;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyString;
use vortex::dtype::FieldName;
use vortex::error::VortexResult;
use vortex::expr::{ExprRef, SelectExpr, root, select};
use vortex::file::{VortexFile, VortexOpenOptions};
use vortex::iter::ArrayIteratorExt;
use vortex::scan::SplitBy;
use vortex::{ArrayRef, ToCanonical};

use crate::arrays::PyArrayRef;
use crate::arrow::{IntoPyArrow, ToPyArrow};
use crate::expr::PyExpr;
use crate::object_store_urls::object_store_from_url;
use crate::{TOKIO_RUNTIME, install_module};

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
    projection: ExprRef,
    filter: Option<ExprRef>,
    indices: Option<ArrayRef>,
) -> VortexResult<ArrayRef> {
    let mut scan = vortex_file.scan()?.with_projection(projection);

    if let Some(filter) = filter {
        scan = scan.with_filter(filter);
    }

    if let Some(indices) = indices {
        let indices = indices.to_primitive()?.into_buffer();
        scan = scan.with_row_indices(indices);
    }

    scan.into_array_iter_multithread()?.read_all()
}

fn projection_from_python(columns: Option<Vec<Bound<PyAny>>>) -> PyResult<ExprRef> {
    fn field_from_pyany(field: &Bound<PyAny>) -> PyResult<FieldName> {
        if field.clone().is_instance_of::<PyString>() {
            Ok(FieldName::from(field.downcast::<PyString>()?.to_str()?))
        } else {
            Err(PyTypeError::new_err(format!(
                "projection: expected list of strings or None, but found: {field}.",
            )))
        }
    }

    Ok(match columns {
        None => root(),
        Some(columns) => {
            let fields = columns
                .iter()
                .map(field_from_pyany)
                .collect::<PyResult<_>>()?;

            SelectExpr::include_expr(fields, root())
        }
    })
}

fn filter_from_python(row_filter: Option<&Bound<PyExpr>>) -> Option<ExprRef> {
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

    pub async fn from_url(url: &str) -> VortexResult<Self> {
        let (_scheme, object_store, path) = object_store_from_url(url)?;
        PyVortexDataset::try_new(
            VortexOpenOptions::file()
                .open_object_store(&object_store, path.as_ref())
                .await?,
        )
    }
}

#[pymethods]
impl PyVortexDataset {
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
        let array = read_array_from_reader(
            &self.vxf,
            projection_from_python(columns)?,
            filter_from_python(row_filter),
            indices.map(|i| i.into_inner()),
        )?;
        Ok(PyArrayRef::from(array))
    }

    #[pyo3(signature = (*, columns = None, row_filter = None, indices = None, split_by = None))]
    pub fn to_record_batch_reader(
        self_: PyRef<Self>,
        columns: Option<Vec<Bound<'_, PyAny>>>,
        row_filter: Option<&Bound<'_, PyExpr>>,
        indices: Option<PyArrayRef>,
        split_by: Option<usize>,
    ) -> PyResult<PyObject> {
        let mut scan = self_
            .vxf
            .scan()?
            .with_projection(projection_from_python(columns)?)
            .with_some_filter(filter_from_python(row_filter))
            .with_split_by(split_by.map(SplitBy::RowCount).unwrap_or(SplitBy::Layout));

        if let Some(indices) = indices.map(|i| i.inner().clone()) {
            let indices = indices.to_primitive()?.into_buffer();
            scan = scan.with_row_indices(indices);
        }

        // TODO(ngates): should we use multi-threaded read or not?
        let schema = Arc::new(scan.dtype()?.to_arrow_schema()?);
        let reader: Box<dyn RecordBatchReader + Send> =
            Box::new(scan.into_record_batch_reader_multithread(schema)?);

        reader.into_pyarrow(self_.py())
    }

    /// The number of rows matching the filter.
    #[pyo3(signature = (*, row_filter = None, split_by = None))]
    pub fn count_rows(
        self_: PyRef<Self>,
        row_filter: Option<&Bound<'_, PyExpr>>,
        split_by: Option<usize>,
    ) -> PyResult<usize> {
        let scan = self_
            .vxf
            .scan()?
            .with_projection(select(vec![], root()))
            .with_some_filter(filter_from_python(row_filter))
            .with_split_by(split_by.map(SplitBy::RowCount).unwrap_or(SplitBy::Layout));

        // TODO(ngates): should we use multi-threaded read or not?
        let schema = Arc::new(scan.dtype()?.to_arrow_schema()?);
        let n_rows: usize = scan
            .into_record_batch_reader_multithread(schema)?
            .map_ok(|rb| rb.num_rows())
            .process_results(|iter| iter.sum())
            .map_err(|err| PyValueError::new_err(format!("arrow error: {}", err)))?;

        Ok(n_rows)
    }
}

#[pyfunction]
pub fn dataset_from_url(url: &str) -> PyResult<PyVortexDataset> {
    Ok(TOKIO_RUNTIME.block_on(PyVortexDataset::from_url(url))?)
}
