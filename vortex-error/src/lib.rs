#![feature(error_generic_member_access)]

#[cfg(feature = "python")]
pub mod python;

use std::backtrace::Backtrace;
use std::borrow::Cow;
use std::fmt::{Debug, Display, Formatter};
use std::ops::Deref;
use std::{env, fmt, io};

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

#[derive(thiserror::Error)]
pub enum VortexError {
    #[error("index {0} out of bounds from {1} to {2}\nBacktrace:\n{3}")]
    OutOfBounds(usize, usize, usize, Backtrace),
    #[error("{0}\nBacktrace:\n{1}")]
    ComputeError(ErrString, Backtrace),
    #[error("{0}\nBacktrace:\n{1}")]
    InvalidArgument(ErrString, Backtrace),
    #[error("{0}\nBacktrace:\n{1}")]
    InvalidSerde(ErrString, Backtrace),
    #[error("function {0} not implemented for {1}\nBacktrace:\n{2}")]
    NotImplemented(ErrString, ErrString, Backtrace),
    #[error("expected type: {0} but instead got {1}\nBacktrace:\n{2}")]
    MismatchedTypes(ErrString, ErrString, Backtrace),
    #[error("{0}\nBacktrace:\n{1}")]
    AssertionFailed(ErrString, Backtrace),
    #[error("{0}: {1}")]
    Context(ErrString, Box<VortexError>),
    #[error(transparent)]
    ArrowError(
        #[from]
        #[backtrace]
        arrow_schema::ArrowError,
    ),
    #[cfg(feature = "flatbuffers")]
    #[error(transparent)]
    FlatBuffersError(
        #[from]
        #[backtrace]
        flatbuffers::InvalidFlatbuffer,
    ),
    #[cfg(feature = "flexbuffers")]
    #[error(transparent)]
    FlexBuffersReaderError(
        #[from]
        #[backtrace]
        flexbuffers::ReaderError,
    ),
    #[cfg(feature = "flexbuffers")]
    #[error(transparent)]
    FlexBuffersDeError(
        #[from]
        #[backtrace]
        flexbuffers::DeserializationError,
    ),
    #[cfg(feature = "flexbuffers")]
    #[error(transparent)]
    FlexBuffersSerError(
        #[from]
        #[backtrace]
        flexbuffers::SerializationError,
    ),
    #[error(transparent)]
    FmtError(
        #[from]
        #[backtrace]
        std::fmt::Error,
    ),
    #[error(transparent)]
    IOError(
        #[from]
        #[backtrace]
        io::Error,
    ),
    #[error(transparent)]
    Utf8Error(
        #[from]
        #[backtrace]
        std::str::Utf8Error,
    ),
    #[cfg(feature = "parquet")]
    #[error(transparent)]
    ParquetError(
        #[from]
        #[backtrace]
        parquet::errors::ParquetError,
    ),
    #[error(transparent)]
    TryFromSliceError(
        #[from]
        #[backtrace]
        std::array::TryFromSliceError,
    ),
    #[cfg(feature = "worker")]
    #[error(transparent)]
    WorkerError(
        #[from]
        #[backtrace]
        worker::Error,
    ),
    #[cfg(feature = "object_store")]
    #[error(transparent)]
    ObjectStore(
        #[from]
        #[backtrace]
        object_store::Error,
    ),
    #[cfg(feature = "datafusion")]
    #[error(transparent)]
    DataFusion(
        #[from]
        #[backtrace]
        datafusion_common::DataFusionError,
    ),
    #[error(transparent)]
    JiffError(
        #[from]
        #[backtrace]
        jiff::Error,
    ),
    #[error(transparent)]
    UrlError(
        #[from]
        #[backtrace]
        url::ParseError,
    ),
}

impl VortexError {
    pub fn with_context<T: Into<ErrString>>(self, msg: T) -> Self {
        VortexError::Context(msg.into(), Box::new(self))
    }
}

impl Debug for VortexError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

pub type VortexResult<T> = Result<T, VortexError>;

pub trait VortexUnwrap {
    type Output;

    fn vortex_unwrap(self) -> Self::Output;
}

impl<T> VortexUnwrap for VortexResult<T> {
    type Output = T;

    #[inline(always)]
    fn vortex_unwrap(self) -> Self::Output {
        self.unwrap_or_else(|err| vortex_panic!(err))
    }
}

pub trait VortexExpect {
    type Output;

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

#[macro_export]
macro_rules! vortex_bail {
    ($($tt:tt)+) => {
        return Err($crate::vortex_err!($($tt)+))
    };
}

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
