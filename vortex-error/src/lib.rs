// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// #![deny(missing_docs)]

//! This crate defines error & result types for Vortex.
//! It also contains a variety of useful macros for error handling.

#[cfg(feature = "python")]
pub mod python;

mod ext;

use std::backtrace::Backtrace;
use std::borrow::Cow;
use std::convert::Infallible;
use std::fmt::{Debug, Display, Formatter};
use std::ops::Deref;
use std::sync::Arc;
use std::{env, fmt};

pub use ext::*;

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

impl From<Infallible> for VortexError {
    fn from(_: Infallible) -> Self {
        unreachable!()
    }
}

error_set::error_set! {
    /// Error parsing third-party formats
    ParsingError = {
        /// Failed to read or write parquet file
        #[cfg(feature = "parquet")]
        #[display("Failed to read parquet file: {0}")]
        ParquetError(parquet::errors::ParquetError),
        /// A wrapper for errors from the FlatBuffers library.
        #[cfg(feature = "flatbuffers")]
        #[display("Failed to parse flatbuffer: {0}")]
        FlatBuffersError(flatbuffers::InvalidFlatbuffer),
        /// Wrap serde and serde json errors
        #[cfg(feature = "serde")]
        #[display("Failed to parse JSON: {0}")]
        SerdeJsonError(serde_json::Error),
        /// A wrapper for UTF-8 conversion errors.
        #[display("Failed to parse convert bytes to UTF-8 string: {0}")]
        Utf8Error(std::str::Utf8Error),
        #[cfg(feature = "prost")]
        /// Wrap errors generated when parsing invalid protobuf messages
        #[display("Buffer doesn't contain a valid protobuff message: {0}")]
        ProstDecodeError(prost::DecodeError),
        /// Wrap prost unknown enum value
        #[cfg(feature = "prost")]
        #[display("Encountered an unknown enum variant value: {0}")]
        ProstUnknownEnumValue(prost::UnknownEnumValue),
        #[cfg(feature = "prost")]
        #[display("Couldn't encode message: {0}")]
        /// Wrap prost encode error
        ProstEncodeError(prost::EncodeError),
        /// A wrapper for URL parsing errors.
        UrlError(url::ParseError),
    };
    /// Io related errors from various sources
    IoError = {
        /// std:io-related errors
        #[display("Io error: {0}")]
        StdIoError(std::io::Error),
        #[cfg(feature = "object_store")]
        #[display("Object storage error: {0}")]
        /// A wrapper for errors from the Object Store library.
        ObjectStore(object_store::Error),
        #[display("{0}")]
        Shared(Arc<VortexError>)
    };
    RuntimeError = {
        #[display("{reason}")]
        InvalidArgument {
            reason: ErrString
        },
        #[display("{reason}")]
        InvalidState {
            reason: ErrString
        },
        #[display("{reason}")]
        InvalidSerde {
            reason: ErrString
        },
        #[display("function {func} not implemented for {by_whom}")]
        NotImplemented {
            func: ErrString,
            by_whom: ErrString
        },
        #[display("expected type: {expected} but instead got: {actual}")]
        MismatchedTypes {
            expected: ErrString,
            actual: ErrString,
        },
        #[display("{reason}")]
        AssertionFailed {
            reason: ErrString
        }
    };
    /// General vortex errors
    GeneralError = {
        /// Out of bounds access
        #[display("index {idx} out of bounds from {start} to {stop}")]
        OutOfBounds {
            idx: usize,
            start: usize,
            stop: usize
        },
        /// Compute error
        #[display("{reason}")]
        ComputeError {
            reason: String
        },
        #[display("Arrow: {0}")]
        ArrowError(arrow_schema::ArrowError),
        TryFromSliceError(std::array::TryFromSliceError),
        JiffError(jiff::Error),
        #[cfg(feature = "tokio")]
        JoinError(tokio::task::JoinError),
        TryFromIntError(std::num::TryFromIntError),
        #[display("{message}: {source}")]
        Context {
            message: ErrString,
            source: Box<VortexError>
        },
    };

    VortexError = ParsingError || IoError || RuntimeError || GeneralError;
}

/// The top-level error type for Vortex.
#[non_exhaustive]
pub enum VortexErrorOld {
    /// An invalid argument was provided.
    InvalidArgument(ErrString, Backtrace),
    /// The system has reached an invalid state,
    InvalidState(ErrString, Backtrace),
    /// An error occurred while serializing or deserializing.
    InvalidSerde(ErrString, Backtrace),
    /// An unimplemented function was called.
    NotImplemented(ErrString, ErrString, Backtrace),
    /// A type mismatch occurred.
    MismatchedTypes(ErrString, ErrString, Backtrace),
    /// An assertion failed.
    AssertionFailed(ErrString, Backtrace),
    /// A wrapper for other errors, carrying additional context.
    Context(ErrString, Box<VortexError>),
    /// A wrapper for errors from the Arrow library.
    ArrowError(arrow_schema::ArrowError, Backtrace),
    /// A wrapper for formatting errors.
    // FmtError(fmt::Error, Backtrace),
    /// A wrapper for IO errors.
    Io(IoError, Backtrace),
}

impl VortexError {
    /// Adds additional context to an error.
    pub fn with_context<T: Into<ErrString>>(self, msg: T) -> Self {
        VortexError::Context {
            message: msg.into(),
            source: self.into(),
        }
    }
}

// impl Display for VortexError {
//     fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
//         use VortexError::*;
//         match self {
//             Generic(err, backtrace) => {
//                 write!(f, "{}\nBacktrace:\n{}", err, backtrace)
//             }
//             InvalidArgument(msg, backtrace) => {
//                 write!(f, "{}\nBacktrace:\n{}", msg, backtrace)
//             }
//             InvalidState(msg, backtrace) => {
//                 write!(f, "{}\nBacktrace:\n{}", msg, backtrace)
//             }
//             InvalidSerde(msg, backtrace) => {
//                 write!(f, "{}\nBacktrace:\n{}", msg, backtrace)
//             }
//             NotImplemented(func, by_whom, backtrace) => {
//                 write!(
//                     f,
//                     "function {} not implemented for {}\nBacktrace:\n{}",
//                     func, by_whom, backtrace
//                 )
//             }
//             MismatchedTypes(expected, actual, backtrace) => {
//                 write!(
//                     f,
//                     "expected type: {} but instead got {}\nBacktrace:\n{}",
//                     expected, actual, backtrace
//                 )
//             }
//             AssertionFailed(msg, backtrace) => {
//                 write!(f, "{}\nBacktrace:\n{}", msg, backtrace)
//             }
//             Context(msg, inner) => {
//                 write!(f, "{}: {}", msg, inner)
//             }
//             ArrowError(err, backtrace) => {
//                 write!(f, "{}\nBacktrace:\n{}", err, backtrace)
//             }
//             #[cfg(feature = "flatbuffers")]
//             // FmtError(err, backtrace) => {
//             //     write!(f, "{}\nBacktrace:\n{}", err, backtrace)
//             // }
//             Io(err, backtrace) => {
//                 write!(f, "{}\nBacktrace:\n{}", err, backtrace)
//             }
//             TryFromSliceError(err, backtrace) => {
//                 write!(f, "{}\nBacktrace:\n{}", err, backtrace)
//             }
//             JiffError(err, backtrace) => {
//                 write!(f, "{}\nBacktrace:\n{}", err, backtrace)
//             }
//             #[cfg(feature = "tokio")]
//             JoinError(err, backtrace) => {
//                 write!(f, "{}\nBacktrace:\n{}", err, backtrace)
//             }
//             TryFromInt(err, backtrace) => {
//                 write!(f, "{}\nBacktrace:\n{}", err, backtrace)
//             }
//             Parsing(err, backtrace) => {
//                 write!(f, "{}\nBacktrace:\n{}", err, backtrace)
//             }
//         }
//     }
// }

// impl Error for VortexError {
//     fn source(&self) -> Option<&(dyn Error + 'static)> {
//         use VortexError::*;
//         match self {
//             Generic(err, _) => Some(err.as_ref()),
//             Context(_, inner) => inner.source(),
//             ArrowError(err, _) => Some(err),
//             // FmtError(err, _) => Some(err),
//             Io(err, _) => Some(err),
//             TryFromSliceError(err, _) => Some(err),
//             JiffError(err, _) => Some(err),
//             #[cfg(feature = "tokio")]
//             JoinError(err, _) => Some(err),
//             TryFromInt(err, _) => Some(err),
//             Parsing(err, _) => Some(err),
//             _ => None,
//         }
//     }
// }

/// A type alias for Results that return VortexErrors as their error type.
pub type VortexResult<T> = Result<T, VortexError>;

/// A vortex result that can be shared or cloned.
pub type SharedVortexResult<T> = Result<T, Arc<VortexError>>;

// impl From<&Arc<VortexError>> for VortexError {
//     fn from(e: &Arc<VortexError>) -> Self {
//         if let VortexError::Shared(e_inner) = e.as_ref() {
//             // don't re-wrap
//             VortexError::Shared(Arc::clone(e_inner))
//         } else {
//             VortexError::Shared(Arc::clone(e))
//         }
//     }
// }

/// A trait for unwrapping a VortexResult.
pub trait VortexUnwrap {
    /// The type of the value being unwrapped.
    type Output;

    /// Returns the value of the result if it is Ok, otherwise panics with the error.
    /// Should be called only in contexts where the error condition represents a bug (programmer error).
    fn vortex_unwrap(self) -> Self::Output;
}

impl<T, E> VortexUnwrap for Result<T, E>
where
    E: Into<VortexError>,
{
    type Output = T;

    #[inline(always)]
    fn vortex_unwrap(self) -> Self::Output {
        self.map_err(|err| err.into())
            .unwrap_or_else(|err| vortex_panic!(err))
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

impl<T, E> VortexExpect for Result<T, E>
where
    E: Into<VortexError>,
{
    type Output = T;

    #[inline(always)]
    fn vortex_expect(self, msg: &str) -> Self::Output {
        match self.map_err(|err| err.into()) {
            Ok(v) => v,
            Err(e) => vortex_panic!(e.with_context(msg.to_string())),
        }
    }
}

impl<T> VortexExpect for Option<T> {
    type Output = T;

    #[inline(always)]
    fn vortex_expect(self, msg: &str) -> Self::Output {
        self.unwrap_or_else(|| {
            let err = RuntimeError::AssertionFailed {
                reason: msg.to_string().into(),
            }
            .into();
            vortex_panic!(err)
        })
    }
}

/// A convenient macro for creating a VortexError.
#[macro_export]
macro_rules! vortex_err {
    (AssertionFailed: $($tts:tt)*) => {{
        use std::backtrace::Backtrace;
        let err_string = format!($($tts)*);
        $crate::__private::must_use(
            $crate::VortexError::AssertionFailed(err_string.into(), Backtrace::capture())
        )
    }};
    (IOError: $($tts:tt)*) => {{
        use std::backtrace::Backtrace;
        $crate::__private::must_use(
            $crate::VortexError::IOError(err_string.into(), Backtrace::capture())
        )
    }};
    (OutOfBounds: $idx:expr, $start:expr, $stop:expr) => {{

        $crate::__private::must_use(
            $crate::VortexError::OutOfBounds { idx: $idx, start: $start, stop: $stop }
        )
    }};
    (NotImplemented: $func:expr, $by_whom:expr) => {{
        $crate::__private::must_use(
            $crate::VortexError::NotImplemented { func: $func.into(), by_whom: format!("{}", $by_whom).into() }
        )
    }};
    (MismatchedTypes: $expected:literal, $actual:expr) => {{
        $crate::__private::must_use(
            $crate::VortexError::MismatchedTypes { expected: $expected.into(), actual: $actual.to_string().into() }
        )
    }};
    (MismatchedTypes: $expected:expr, $actual:expr) => {{

        $crate::__private::must_use(
            $crate::VortexError::MismatchedTypes { expected: $expected.to_string().into(), actual: $actual.to_string().into() }
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
            $crate::VortexError::$variant { reason: format!($fmt, $($arg),*).into() }
        )
    }};
    // ($variant:ident: $err:expr $(,)?) => {
    //     $crate::__private::must_use(
    //         $crate::VortexError::$variant($err)
    //     )
    // };
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
