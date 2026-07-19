# pyo3-object_store

Integration between [`object_store`](https://docs.rs/object_store) and [`pyo3`](https://github.com/PyO3/pyo3).

This provides Python builder classes so that Python users can easily create [`Arc<dyn ObjectStore>`][object_store::ObjectStore] instances, which can then be used in pure-Rust code.

## Usage

1. Register the builders.

    ```rs
    #[pymodule]
    fn python_module(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
        pyo3_object_store::register_store_module(py, m, "python_module", "store")?;
        pyo3_object_store::register_exceptions_module(py, m, "python_module", "exceptions")?;
    }
    ```

   This exports the underlying Python classes from your own Rust-Python library.

   Refer to [`register_store_module`] and [`register_exceptions_module`] for more information.

2. Accept [`PyObjectStore`] as a parameter in your function exported to Python. Its [`into_dyn`][PyObjectStore::into_dyn] method (or `Into` impl) gives you an [`Arc<dyn ObjectStore>`][object_store::ObjectStore].

    ```rs
    #[pyfunction]
    pub fn use_object_store(store: PyObjectStore) {
        let store: Arc<dyn ObjectStore> = store.into_dyn();
    }
    ```

   You can also accept [`AnyObjectStore`] as a parameter, which wraps [`PyObjectStore`] and [`PyExternalObjectStore`]. This allows you to seamlessly recreate `ObjectStore` instances that users pass in from other Python libraries (like [`obstore`][obstore]) that themselves export `pyo3-object_store` builders.

   Note however that due to lack of [ABI stability](#abi-stability), `ObjectStore` instances will be **recreated**, and so there will be no connection pooling across the external store.

## Example

The [`obstore`][obstore] Python library gives a full real-world example of using `pyo3-object_store`, exporting a Python API that mimics the Rust [`ObjectStore`][object_store::ObjectStore] API.

[obstore]: https://developmentseed.org/obstore/latest/

## Feature flags

- `external-store-warning` (enabled by default): Emit a user warning when constructing a `PyExternalObjectStore` (or `AnyObjectStore::PyExternalObjectStore`), to inform users that there may be performance implications due to lack of connection pooling across separately-compiled Python libraries. Disable this feature if you don't want the warning.

## ABI stability

It's [not currently possible](https://github.com/PyO3/pyo3/issues/1444) to share a `#[pyclass]` across multiple Python libraries, except in special cases where the underlying data has a stable ABI.

As `object_store` does not currently have a stable ABI, we can't share `PyObjectStore` instances across multiple separately-compiled Python libraries.

We have two ways to get around this:

- Export your own Python classes so that users can construct `ObjectStore` instances that were compiled _with your library_. See [`register_store_module`].
- Accept [`AnyObjectStore`] or [`PyExternalObjectStore`] as a parameter, which allows for seamlessly **reconstructing** stores from an external Python library, like [`obstore`][obstore]. This has some overhead and removes any possibility of connection pooling across the two instances.

Note about not being able to use these across Python packages. It has to be used with the exported classes from your own library.

## Python Type hints

We don't yet have a _great_ solution here for reusing the store builder type hints in your own library. Type hints are shipped with the cargo dependency. Or, you can use a submodule on the `obstore` repo. See [`async-tiff` for an example](https://github.com/developmentseed/async-tiff/blob/35eaf116d9b1ab31232a1e23298b3102d2879e9c/python/python/async_tiff/store).

## Version compatibility

| pyo3-object_store | pyo3               | object_store       |
| ----------------- | ------------------ | ------------------ |
| 0.1.x             | 0.23               | 0.12               |
| 0.2.x             | 0.24               | 0.12               |
| 0.3.x             | **0.23** :warning: | **0.11** :warning: |
| 0.4.x             | 0.24               | **0.11** :warning: |
| 0.5.x             | 0.25               | 0.12               |
| 0.6.x             | 0.26               | 0.12               |
| 0.7.x             | 0.27               | 0.12               |
| 0.8.x             | 0.27               | 0.13               |
| 0.9.x             | 0.28               | 0.13               |
| 0.10.x            | 0.28               | **0.12** :warning: |
| 0.11.x            | 0.29               | 0.13               |

Note that 0.3.x, 0.4.x, and 0.10.x are compatibility releases to use `pyo3-object_store` with older versions of `pyo3` and `object_store`.
