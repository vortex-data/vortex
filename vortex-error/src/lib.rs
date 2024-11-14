#![feature(error_generic_member_access)]
#![deny(missing_docs)]

//! This crate defines error & result types for Vortex.
//! It also contains a variety of useful macros for error handling.

#[cfg(feature = "python")]
pub mod python;

use std::backtrace::Backtrace;
use std::borrow::Cow;
use std::fmt::{Debug, Display, Formatter};
use std::ops::Deref;
use std::{env, fmt, io};

/// A string that can be used as an error message.
#[derive(Debug)]
pub struct ErrString(Cow<'static, str>);

#[allow(clippy::fallible_impl_from)]
impl<T> From<T> for ErrString
where
    T: Into<Cow<'static, str>>,
{
    #[allow(clippy::panic)]
    fn from(msg: T) -> Self {
        if env::var("VORTEX_PANIC_ON_ERR").as_deref().unwrap_or("") == "1" {
            panic!("{}\nBacktrace:\n{}", msg.into(), Backtrace::capture());
        } else {
            Self(msg.into())
        }
    }
}

impl AsRef<str> for ErrString {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Deref for ErrString {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Display for ErrString {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

/// The top-level error type for Vortex.
#[derive(thiserror::Error)]
#[non_exhaustive]
pub enum VortexError {
    /// An index is out of bounds.
    #[error("index {0} out of bounds from {1} to {2}\nBacktrace:\n{3}")]
    OutOfBounds(usize, usize, usize, Backtrace),
    /// An error occurred while executing a compute kernel.
    #[error("{0}\nBacktrace:\n{1}")]
    ComputeError(ErrString, Backtrace),
    /// An invalid argument was provided.
    #[error("{0}\nBacktrace:\n{1}")]
    InvalidArgument(ErrString, Backtrace),
    /// An error occurred while serializing or deserializing.
    #[error("{0}\nBacktrace:\n{1}")]
    InvalidSerde(ErrString, Backtrace),
    /// An unimplemented function was called.
    #[error("function {0} not implemented for {1}\nBacktrace:\n{2}")]
    NotImplemented(ErrString, ErrString, Backtrace),
    /// A type mismatch occurred.
    #[error("expected type: {0} but instead got {1}\nBacktrace:\n{2}")]
    MismatchedTypes(ErrString, ErrString, Backtrace),
    /// An assertion failed.
    #[error("{0}\nBacktrace:\n{1}")]
    AssertionFailed(ErrString, Backtrace),
    /// A wrapper for other errors, carrying additional context.
    #[error("{0}: {1}")]
    Context(ErrString, #[source] Box<VortexError>),
    /// A wrapper for errors from the Arrow library.
    #[error(transparent)]
    ArrowError(
        #[from]
        #[backtrace]
        arrow_schema::ArrowError,
    ),
    /// A wrapper for errors from the FlatBuffers library.
    #[cfg(feature = "flatbuffers")]
    #[error(transparent)]
    FlatBuffersError(
        #[from]
        #[backtrace]
        flatbuffers::InvalidFlatbuffer,
    ),
    /// A wrapper for reader errors from the FlexBuffers library.
    #[cfg(feature = "flexbuffers")]
    #[error(transparent)]
    FlexBuffersReaderError(
        #[from]
        #[backtrace]
        flexbuffers::ReaderError,
    ),
    /// A wrapper for deserialization errors from the FlexBuffers library.
    #[cfg(feature = "flexbuffers")]
    #[error(transparent)]
    FlexBuffersDeError(
        #[from]
        #[backtrace]
        flexbuffers::DeserializationError,
    ),
    /// A wrapper for serialization errors from the FlexBuffers library.
    #[cfg(feature = "flexbuffers")]
    #[error(transparent)]
    FlexBuffersSerError(
        #[from]
        #[backtrace]
        flexbuffers::SerializationError,
    ),
    /// A wrapper for formatting errors.
    #[error(transparent)]
    FmtError(
        #[from]
        #[backtrace]
        fmt::Error,
    ),
    /// A wrapper for IO errors.
    #[error(transparent)]
    IOError(
        #[from]
        #[backtrace]
        io::Error,
    ),
    /// A wrapper for UTF-8 conversion errors.
    #[error(transparent)]
    Utf8Error(
        #[from]
        #[backtrace]
        std::str::Utf8Error,
    ),
    /// A wrapper for errors from the Parquet library.
    #[cfg(feature = "parquet")]
    #[error(transparent)]
    ParquetError(
        #[from]
        #[backtrace]
        parquet::errors::ParquetError,
    ),
    /// A wrapper for errors from the standard library when converting a slice to an array.
    #[error(transparent)]
    TryFromSliceError(
        #[from]
        #[backtrace]
        std::array::TryFromSliceError,
    ),
    /// A wrapper for errors from the Cloudflare Workers library.
    #[cfg(feature = "worker")]
    #[error(transparent)]
    WorkerError(
        #[from]
        #[backtrace]
        worker::Error,
    ),
    /// A wrapper for errors from the Object Store library.
    #[cfg(feature = "object_store")]
    #[error(transparent)]
    ObjectStore(
        #[from]
        #[backtrace]
        object_store::Error,
    ),
    /// A wrapper for errors from DataFusion.
    #[cfg(feature = "datafusion")]
    #[error(transparent)]
    DataFusion(
        #[from]
        #[backtrace]
        datafusion_common::DataFusionError,
    ),
    /// A wrapper for errors from the Jiff library.
    #[error(transparent)]
    JiffError(
        #[from]
        #[backtrace]
        jiff::Error,
    ),
    /// A wrapper for URL parsing errors.
    #[error(transparent)]
    UrlError(
        #[from]
        #[backtrace]
        url::ParseError,
    ),
}

impl VortexError {
    /// Adds additional context to an error.
    pub fn with_context<T: Into<ErrString>>(self, msg: T) -> Self {
        VortexError::Context(msg.into(), Box::new(self))
    }
}

impl Debug for VortexError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

/// A type alias for Results that return VortexErrors as their error type.
pub type VortexResult<T> = Result<T, VortexError>;

/// A trait for unwrapping a VortexResult.
pub trait VortexUnwrap {
    /// The type of the value being unwrapped.
    type Output;

    /// Returns the value of the result if it is Ok, otherwise panics with the error.
    /// Should be called only in contexts where the error condition represents a bug (programmer error).
    fn vortex_unwrap(self) -> Self::Output;
}

impl<T> VortexUnwrap for VortexResult<T> {
    type Output = T;

    #[inline(always)]
    fn vortex_unwrap(self) -> Self::Output {
        self.unwrap_or_else(|err| vortex_panic!(err))
    }
}

/// A trait for expect-ing a VortexResult or an Option.
pub trait VortexExpect {
    /// The type of the value being expected.
    type Output;

    /// Returns the value of the result if it is Ok, otherwise panics with the error.
    /// Should be called only in contexts where the error condition represents a bug (programmer error).
    fn vortex_expect(self, msg: &str) -> Self::Output;
}

impl<T> VortexExpect for VortexResult<T> {
    type Output = T;

    #[inline(always)]
    fn vortex_expect(self, msg: &str) -> Self::Output {
        self.unwrap_or_else(|e| vortex_panic!(e.with_context(msg.to_string())))
    }
}

impl<T> VortexExpect for Option<T> {
    type Output = T;

    #[inline(always)]
    fn vortex_expect(self, msg: &str) -> Self::Output {
        self.unwrap_or_else(|| {
            let err = VortexError::AssertionFailed(msg.to_string().into(), Backtrace::capture());
            vortex_panic!(err)
        })
    }
}

/// A convenient macro for creating a VortexError.
#[macro_export]
macro_rules! vortex_err {
    (OutOfBounds: $idx:expr, $start:expr, $stop:expr) => {{
        use std::backtrace::Backtrace;
        $crate::__private::must_use(
            $crate::VortexError::OutOfBounds($idx, $start, $stop, Backtrace::capture())
        )
    }};
    (NotImplemented: $func:expr, $by_whom:expr) => {{
        use std::backtrace::Backtrace;
        $crate::__private::must_use(
            $crate::VortexError::NotImplemented($func.into(), format!("{}", $by_whom).into(), Backtrace::capture())
        )
    }};
    (MismatchedTypes: $expected:literal, $actual:expr) => {{
        use std::backtrace::Backtrace;
        $crate::__private::must_use(
            $crate::VortexError::MismatchedTypes($expected.into(), $actual.to_string().into(), Backtrace::capture())
        )
    }};
    (MismatchedTypes: $expected:expr, $actual:expr) => {{
        use std::backtrace::Backtrace;
        $crate::__private::must_use(
            $crate::VortexError::MismatchedTypes($expected.to_string().into(), $actual.to_string().into(), Backtrace::capture())
        )
    }};
    (Context: $msg:literal, $err:expr) => {{
        $crate::__private::must_use(
            $crate::VortexError::Context($msg.into(), Box::new($err))
        )
    }};
    ($variant:ident: $fmt:literal $(, $arg:expr)* $(,)?) => {{
        use std::backtrace::Backtrace;
        $crate::__private::must_use(
            $crate::VortexError::$variant(format!($fmt, $($arg),*).into(), Backtrace::capture())
        )
    }};
    ($variant:ident: $err:expr $(,)?) => {
        $crate::__private::must_use(
            $crate::VortexError::$variant($err)
        )
    };
    ($fmt:literal $(, $arg:expr)* $(,)?) => {
        $crate::vortex_err!(InvalidArgument: $fmt, $($arg),*)
    };
}

/// A convenient macro for returning a VortexError.
#[macro_export]
macro_rules! vortex_bail {
    ($($tt:tt)+) => {
        return Err($crate::vortex_err!($($tt)+))
    };
}

/// A convenient macro for panicking with a VortexError in the presence of a programmer error
/// (e.g., an invariant has been violated).
#[macro_export]
macro_rules! vortex_panic {
    (OutOfBounds: $idx:expr, $start:expr, $stop:expr) => {{
        $crate::vortex_panic!($crate::vortex_err!(OutOfBounds: $idx, $start, $stop))
    }};
    (NotImplemented: $func:expr, $for_whom:expr) => {{
        $crate::vortex_panic!($crate::vortex_err!(NotImplemented: $func, $for_whom))
    }};
    (MismatchedTypes: $expected:literal, $actual:expr) => {{
        $crate::vortex_panic!($crate::vortex_err!(MismatchedTypes: $expected, $actual))
    }};
    (MismatchedTypes: $expected:expr, $actual:expr) => {{
        $crate::vortex_panic!($crate::vortex_err!(MismatchedTypes: $expected, $actual))
    }};
    (Context: $msg:literal, $err:expr) => {{
        $crate::vortex_panic!($crate::vortex_err!(Context: $msg, $err))
    }};
    ($variant:ident: $fmt:literal $(, $arg:expr)* $(,)?) => {
        $crate::vortex_panic!($crate::vortex_err!($variant: $fmt, $($arg),*))
    };
    ($err:expr, $fmt:literal $(, $arg:expr)* $(,)?) => {{
        let err: $crate::VortexError = $err;
        panic!("{}", err.with_context(format!($fmt, $($arg),*)))
    }};
    ($fmt:literal $(, $arg:expr)* $(,)?) => {
        $crate::vortex_panic!($crate::vortex_err!($fmt, $($arg),*))
    };
    ($err:expr) => {{
        let err: $crate::VortexError = $err;
        panic!("{}", err)
    }};
}

#[cfg(feature = "datafusion")]
impl From<VortexError> for datafusion_common::DataFusionError {
    fn from(value: VortexError) -> Self {
        Self::External(Box::new(value))
    }
}

#[cfg(feature = "datafusion")]
impl From<VortexError> for datafusion_common::arrow::error::ArrowError {
    fn from(value: VortexError) -> Self {
        match value {
            VortexError::ArrowError(e) => e,
            _ => Self::from_external_error(Box::new(value)),
        }
    }
}

// Not public, referenced by macros only.
#[doc(hidden)]
pub mod __private {
    #[doc(hidden)]
    #[inline]
    #[cold]
    #[must_use]
    pub const fn must_use(error: crate::VortexError) -> crate::VortexError {
        error
    }
}

#[cfg(feature = "worker")]
impl From<VortexError> for worker::Error {
    fn from(value: VortexError) -> Self {
        Self::RustError(value.to_string())
    }
}
