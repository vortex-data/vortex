use pyo3::intern;
use pyo3::prelude::*;

use crate::error::*;
use crate::{
    from_url, PyAzureStore, PyGCSStore, PyHttpStore, PyLocalStore, PyMemoryStore, PyS3Store,
};

/// Export the default Python API as a submodule named `store` within the given parent module
///
/// The following will add a `store` submodule within a Python top-level module called `"python_module"`.
///
/// Args:
///
/// - [`Python`][pyo3::prelude::Python] token
/// - parent_module: [`PyModule`][pyo3::prelude::PyModule] object
/// - parent_module_str: the string name of the parent Python module for how this is exported.
/// - sub_module_str: the string name of **this** Python module. Usually `store` but `_store` may be preferred in some cases.
///
/// ```notest
/// #[pymodule]
/// fn rust_module(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
///     pyo3_object_store::register_store_module(py, m, "python_module")?;
/// }
/// ```
///
/// Or as another example, in the `obstore` Python-facing API, this is exported as
///
/// ```notest
/// #[pymodule]
/// fn _obstore(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
///     pyo3_object_store::register_store_module(py, m, "obstore")?;
/// }
/// ```
///
/// Then `obstore._obstore.*` is re-exported at the top-level at `obstore.*`. So this means the
/// store will be available at `obstore.store`.
///
// https://github.com/PyO3/pyo3/issues/1517#issuecomment-808664021
// https://github.com/PyO3/pyo3/issues/759#issuecomment-977835119
pub fn register_store_module(
    py: Python<'_>,
    parent_module: &Bound<'_, PyModule>,
    parent_module_str: &str,
    sub_module_str: &str,
) -> PyResult<()> {
    let full_module_string = format!("{parent_module_str}.{sub_module_str}");

    let child_module = PyModule::new(parent_module.py(), sub_module_str)?;

    child_module.add_wrapped(wrap_pyfunction!(from_url))?;
    child_module.add_class::<PyAzureStore>()?;
    child_module.add_class::<PyGCSStore>()?;
    child_module.add_class::<PyHttpStore>()?;
    child_module.add_class::<PyLocalStore>()?;
    child_module.add_class::<PyMemoryStore>()?;
    child_module.add_class::<PyS3Store>()?;

    // Set the value of `__module__` correctly on each publicly exposed function or class
    let __module__ = intern!(py, "__module__");
    child_module
        .getattr("from_url")?
        .setattr(__module__, &full_module_string)?;
    child_module
        .getattr("AzureStore")?
        .setattr(__module__, &full_module_string)?;
    child_module
        .getattr("GCSStore")?
        .setattr(__module__, &full_module_string)?;
    child_module
        .getattr("HTTPStore")?
        .setattr(__module__, &full_module_string)?;
    child_module
        .getattr("LocalStore")?
        .setattr(__module__, &full_module_string)?;
    child_module
        .getattr("MemoryStore")?
        .setattr(__module__, &full_module_string)?;
    child_module
        .getattr("S3Store")?
        .setattr(__module__, &full_module_string)?;

    // Add the child module to the parent module
    parent_module.add_submodule(&child_module)?;

    py.import(intern!(py, "sys"))?
        .getattr(intern!(py, "modules"))?
        .set_item(&full_module_string, &child_module)?;

    // needs to be set *after* `add_submodule()`
    child_module.setattr("__name__", full_module_string)?;

    Ok(())
}

/// Export exceptions as a submodule named `exceptions` within the given parent module
// https://github.com/PyO3/pyo3/issues/1517#issuecomment-808664021
// https://github.com/PyO3/pyo3/issues/759#issuecomment-977835119
pub fn register_exceptions_module(
    py: Python<'_>,
    parent_module: &Bound<'_, PyModule>,
    parent_module_str: &str,
    sub_module_str: &str,
) -> PyResult<()> {
    let full_module_string = format!("{parent_module_str}.{sub_module_str}");

    let child_module = PyModule::new(parent_module.py(), sub_module_str)?;

    child_module.add("BaseError", py.get_type::<BaseError>())?;
    child_module.add("GenericError", py.get_type::<GenericError>())?;
    child_module.add("NotFoundError", py.get_type::<NotFoundError>())?;
    child_module.add("InvalidPathError", py.get_type::<InvalidPathError>())?;
    child_module.add("JoinError", py.get_type::<JoinError>())?;
    child_module.add("NotSupportedError", py.get_type::<NotSupportedError>())?;
    child_module.add("AlreadyExistsError", py.get_type::<AlreadyExistsError>())?;
    child_module.add("PreconditionError", py.get_type::<PreconditionError>())?;
    child_module.add("NotModifiedError", py.get_type::<NotModifiedError>())?;
    child_module.add(
        "PermissionDeniedError",
        py.get_type::<PermissionDeniedError>(),
    )?;
    child_module.add(
        "UnauthenticatedError",
        py.get_type::<UnauthenticatedError>(),
    )?;
    child_module.add(
        "UnknownConfigurationKeyError",
        py.get_type::<UnknownConfigurationKeyError>(),
    )?;

    // Set the value of `__module__` correctly on each publicly exposed function or class
    let __module__ = intern!(py, "__module__");
    child_module
        .getattr("BaseError")?
        .setattr(__module__, &full_module_string)?;
    child_module
        .getattr("GenericError")?
        .setattr(__module__, &full_module_string)?;
    child_module
        .getattr("NotFoundError")?
        .setattr(__module__, &full_module_string)?;
    child_module
        .getattr("InvalidPathError")?
        .setattr(__module__, &full_module_string)?;
    child_module
        .getattr("JoinError")?
        .setattr(__module__, &full_module_string)?;
    child_module
        .getattr("NotSupportedError")?
        .setattr(__module__, &full_module_string)?;
    child_module
        .getattr("AlreadyExistsError")?
        .setattr(__module__, &full_module_string)?;
    child_module
        .getattr("PreconditionError")?
        .setattr(__module__, &full_module_string)?;
    child_module
        .getattr("NotModifiedError")?
        .setattr(__module__, &full_module_string)?;
    child_module
        .getattr("PermissionDeniedError")?
        .setattr(__module__, &full_module_string)?;
    child_module
        .getattr("UnauthenticatedError")?
        .setattr(__module__, &full_module_string)?;
    child_module
        .getattr("UnknownConfigurationKeyError")?
        .setattr(__module__, &full_module_string)?;

    // Add the child module to the parent module
    parent_module.add_submodule(&child_module)?;

    py.import(intern!(py, "sys"))?
        .getattr(intern!(py, "modules"))?
        .set_item(full_module_string.as_str(), &child_module)?;

    // needs to be set *after* `add_submodule()`
    child_module.setattr("__name__", full_module_string)?;

    Ok(())
}
