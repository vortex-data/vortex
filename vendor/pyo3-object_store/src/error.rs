//! Contains the [`PyObjectStoreError`], the error enum returned by all fallible functions in this
//! crate.

use pyo3::exceptions::{PyFileNotFoundError, PyIOError, PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::{create_exception, CastError};
use thiserror::Error;

// Base exception
// Note that this is named `BaseError` instead of `ObstoreError` to not leak the name "obstore" to
// other Rust-Python libraries using pyo3-object_store.
create_exception!(
    pyo3_object_store,
    BaseError,
    pyo3::exceptions::PyException,
    "The base Python-facing exception from which all other errors subclass."
);

// Subclasses from base exception
create_exception!(
    pyo3_object_store,
    GenericError,
    BaseError,
    "A Python-facing exception wrapping [object_store::Error::Generic]."
);
create_exception!(
    pyo3_object_store,
    NotFoundError,
    BaseError,
    "A Python-facing exception wrapping [object_store::Error::NotFound]."
);
create_exception!(
    pyo3_object_store,
    InvalidPathError,
    BaseError,
    "A Python-facing exception wrapping [object_store::Error::InvalidPath]."
);
create_exception!(
    pyo3_object_store,
    JoinError,
    BaseError,
    "A Python-facing exception wrapping [object_store::Error::JoinError]."
);
create_exception!(
    pyo3_object_store,
    NotSupportedError,
    BaseError,
    "A Python-facing exception wrapping [object_store::Error::NotSupported]."
);
create_exception!(
    pyo3_object_store,
    AlreadyExistsError,
    BaseError,
    "A Python-facing exception wrapping [object_store::Error::AlreadyExists]."
);
create_exception!(
    pyo3_object_store,
    PreconditionError,
    BaseError,
    "A Python-facing exception wrapping [object_store::Error::Precondition]."
);
create_exception!(
    pyo3_object_store,
    NotModifiedError,
    BaseError,
    "A Python-facing exception wrapping [object_store::Error::NotModified]."
);
create_exception!(
    pyo3_object_store,
    PermissionDeniedError,
    BaseError,
    "A Python-facing exception wrapping [object_store::Error::PermissionDenied]."
);
create_exception!(
    pyo3_object_store,
    UnauthenticatedError,
    BaseError,
    "A Python-facing exception wrapping [object_store::Error::Unauthenticated]."
);
create_exception!(
    pyo3_object_store,
    UnknownConfigurationKeyError,
    BaseError,
    "A Python-facing exception wrapping [object_store::Error::UnknownConfigurationKey]."
);

/// The Error variants returned by this crate.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum PyObjectStoreError {
    /// A wrapped [object_store::Error]
    #[error(transparent)]
    ObjectStoreError(#[from] object_store::Error),

    /// A wrapped [PyErr]
    #[error(transparent)]
    PyErr(#[from] PyErr),

    /// A wrapped [std::io::Error]
    #[error(transparent)]
    IOError(#[from] std::io::Error),
}

impl From<PyObjectStoreError> for PyErr {
    fn from(error: PyObjectStoreError) -> Self {
        match error {
            PyObjectStoreError::PyErr(err) => err,
            PyObjectStoreError::ObjectStoreError(ref err) => match err {
                object_store::Error::Generic {
                    store: _,
                    source: _,
                } => GenericError::new_err(print_with_debug(err)),
                object_store::Error::NotFound { path: _, source: _ } => {
                    PyFileNotFoundError::new_err(print_with_debug(err))
                }
                object_store::Error::InvalidPath { source: _ } => {
                    InvalidPathError::new_err(print_with_debug(err))
                }
                object_store::Error::JoinError { source: _ } => {
                    JoinError::new_err(print_with_debug(err))
                }
                object_store::Error::NotSupported { source: _ } => {
                    NotSupportedError::new_err(print_with_debug(err))
                }
                object_store::Error::AlreadyExists { path: _, source: _ } => {
                    AlreadyExistsError::new_err(print_with_debug(err))
                }
                object_store::Error::Precondition { path: _, source: _ } => {
                    PreconditionError::new_err(print_with_debug(err))
                }
                object_store::Error::NotModified { path: _, source: _ } => {
                    NotModifiedError::new_err(print_with_debug(err))
                }
                object_store::Error::NotImplemented {
                    operation: _,
                    implementer: _,
                } => PyNotImplementedError::new_err(print_with_debug(err)),
                object_store::Error::PermissionDenied { path: _, source: _ } => {
                    PermissionDeniedError::new_err(print_with_debug(err))
                }
                object_store::Error::Unauthenticated { path: _, source: _ } => {
                    UnauthenticatedError::new_err(print_with_debug(err))
                }
                object_store::Error::UnknownConfigurationKey { store: _, key: _ } => {
                    UnknownConfigurationKeyError::new_err(print_with_debug(err))
                }
                _ => GenericError::new_err(print_with_debug(err)),
            },
            PyObjectStoreError::IOError(err) => PyIOError::new_err(err),
        }
    }
}

fn print_with_debug(err: &object_store::Error) -> String {
    // #? gives "pretty-printing" for debug
    // https://doc.rust-lang.org/std/fmt/trait.Debug.html
    format!("{err}\n\nDebug source:\n{err:#?}")
}

impl<'a, 'py> From<CastError<'a, 'py>> for PyObjectStoreError {
    fn from(other: CastError<'a, 'py>) -> Self {
        Self::PyErr(PyValueError::new_err(format!(
            "Could not downcast: {other}",
        )))
    }
}

/// A type wrapper around `Result<T, PyObjectStoreError>`.
pub type PyObjectStoreResult<T> = Result<T, PyObjectStoreError>;

/// A specialized `Error` for object store-related errors
///
/// Vendored from upstream to handle our vendored URL parsing
#[derive(Debug, thiserror::Error)]
pub(crate) enum ParseUrlError {
    #[error(
        "Unknown url scheme cannot be parsed into storage location: {}",
        scheme
    )]
    UnknownUrlScheme { scheme: String },

    #[error("URL did not match any known pattern for scheme: {}", url)]
    UrlNotRecognised { url: String },
}

impl From<ParseUrlError> for object_store::Error {
    fn from(source: ParseUrlError) -> Self {
        Self::Generic {
            store: "S3",
            source: Box::new(source),
        }
    }
}
