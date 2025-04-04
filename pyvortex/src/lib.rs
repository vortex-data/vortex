#![allow(unsafe_op_in_unsafe_fn)]

use std::ops::Deref;
use std::sync::LazyLock;

use axum::http::StatusCode;
use axum::response::IntoResponse;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

pub(crate) mod arrays;
mod compress;
mod dataset;
pub(crate) mod dtype;
mod expr;
mod file;
mod io;
mod iter;
mod object_store_urls;
mod python_repr;
mod record_batch_reader;
mod registry;
pub(crate) mod scalar;
mod serde;

use log::LevelFilter;
use pyo3_log::{Caching, Logger};
use tokio::runtime::Runtime;
use vortex::error::{VortexError, VortexExpect as _};

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[allow(non_upper_case_globals)]
#[export_name = "malloc_conf"]
pub static malloc_conf: &[u8] = b"prof:true,prof_active:true,lg_prof_sample:19\0";

pub static TOKIO_RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    Runtime::new()
        .map_err(VortexError::IOError)
        .vortex_expect("tokio runtime must not fail to start")
});

/// Vortex is an Apache Arrow-compatible toolkit for working with compressed array data.
#[pymodule]
fn _lib(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    Python::with_gil(|py| -> PyResult<()> {
        Logger::new(py, Caching::LoggersAndLevels)?
            .filter(LevelFilter::Info)
            .filter_target("my_module::verbose_submodule".to_owned(), LevelFilter::Warn)
            .install()
            .map(|_| ())
            .map_err(|err| PyRuntimeError::new_err(format!("could not initialize logger {}", err)))
    })?;

    // Initialize our submodules, living under vortex._lib
    arrays::init(py, m)?;
    compress::init(py, m)?;
    dataset::init(py, m)?;
    dtype::init(py, m)?;
    expr::init(py, m)?;
    file::init(py, m)?;
    io::init(py, m)?;
    iter::init(py, m)?;
    registry::init(py, m)?;
    scalar::init(py, m)?;
    serde::init(py, m)?;

    let app = axum::Router::new()
        .route("/debug/pprof/heap", axum::routing::get(handle_get_heap))
        .route(
            "/debug/pprof/heap/flamegraph",
            axum::routing::get(handle_get_heap_flamegraph),
        );

    TOKIO_RUNTIME.spawn(async move {
        // run our app with hyper, listening globally on port 3000
        let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
        axum::serve(listener, app).await.unwrap();
    });

    Ok(())
}

pub async fn handle_get_heap() -> Result<impl IntoResponse, (StatusCode, String)> {
    let mut prof_ctl = jemalloc_pprof::PROF_CTL.as_ref().unwrap().lock().await;
    require_profiling_activated(&prof_ctl)?;
    let pprof = prof_ctl
        .dump_pprof()
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
    Ok(pprof)
}

pub async fn handle_get_heap_flamegraph() -> Result<impl IntoResponse, (StatusCode, String)> {
    use axum::body::Body;
    use axum::http::header::CONTENT_TYPE;
    use axum::response::Response;

    let mut prof_ctl = jemalloc_pprof::PROF_CTL.as_ref().unwrap().lock().await;
    require_profiling_activated(&prof_ctl)?;
    let svg = prof_ctl
        .dump_flamegraph()
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
    Response::builder()
        .header(CONTENT_TYPE, "image/svg+xml")
        .body(Body::from(svg))
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))
}

/// Checks whether jemalloc profiling is activated an returns an error response if not.
fn require_profiling_activated(
    prof_ctl: &jemalloc_pprof::JemallocProfCtl,
) -> Result<(), (StatusCode, String)> {
    if prof_ctl.activated() {
        Ok(())
    } else {
        Err((
            axum::http::StatusCode::FORBIDDEN,
            "heap profiling not activated".into(),
        ))
    }
}

/// Initialize a module and add it to `sys.modules`.
///
/// Without this, it's not possible to use native submodules as "packages". For example:
///
/// ```pycon
/// >>> from vortex._lib.dtype import bool_  # This fails
/// ModuleNotFoundError: No module named 'vortex._lib.dtype'; 'vortex._lib' is not a package
/// ```
///
/// After this, we can import submodules both as modules:
///
/// ```pycon
/// >>> from vortex._lib import dtype
/// ```
///
/// And have direct import access to functions and classes in the submodule:
///
/// ```pycon
/// >>> from vortex._lib.dtype import bool_
/// ```
///
/// See <https://github.com/PyO3/pyo3/issues/759#issuecomment-1811992321>.
pub fn install_module(name: &str, module: &Bound<PyModule>) -> PyResult<()> {
    module
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item(name, module)?;
    // needs to be set *after* `add_submodule()`
    module.setattr("__name__", name)?;
    Ok(())
}

/// An adapter struct used to localize trait impls to this crate.
pub struct PyVortex<T>(pub T);

impl<T> From<T> for PyVortex<T> {
    fn from(value: T) -> Self {
        Self(value)
    }
}

impl<T> PyVortex<T> {
    pub fn into_inner(self) -> T {
        self.0
    }

    pub fn inner(&self) -> &T {
        &self.0
    }
}

impl<T> Deref for PyVortex<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
