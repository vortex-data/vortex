// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]

//! This crate defines error & result types for Vortex.
//! It also contains a variety of useful macros for error handling.

#[cfg(feature = "python")]
pub mod python;

use std::backtrace::Backtrace;
use std::borrow::Cow;
use std::convert::Infallible;
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::num::TryFromIntError;
use std::ops::Deref;
use std::sync::{Arc, PoisonError};
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

impl From<Infallible> for VortexError {
    fn from(_: Infallible) -> Self {
        unreachable!()
    }
}

const _: () = {
    assert!(size_of::<VortexError>() < 128);
};
/// The top-level error type for Vortex.
#[non_exhaustive]
pub enum VortexError {
    /// A wrapped generic error
    Generic(Box<dyn Error + Send + Sync + 'static>, Box<Backtrace>),
    /// An index is out of bounds.
    OutOfBounds(usize, usize, usize, Box<Backtrace>),
    /// An error occurred while executing a compute kernel.
    ComputeError(ErrString, Box<Backtrace>),
    /// An invalid argument was provided.
    InvalidArgument(ErrString, Box<Backtrace>),
    /// The system has reached an invalid state,
    InvalidState(ErrString, Box<Backtrace>),
    /// An error occurred while serializing or deserializing.
    InvalidSerde(ErrString, Box<Backtrace>),
    /// An unimplemented function was called.
    NotImplemented(ErrString, ErrString, Box<Backtrace>),
    /// A type mismatch occurred.
    MismatchedTypes(ErrString, ErrString, Box<Backtrace>),
    /// An assertion failed.
    AssertionFailed(ErrString, Box<Backtrace>),
    /// A wrapper for other errors, carrying additional context.
    Context(ErrString, Box<VortexError>),
    /// A wrapper for shared errors that require cloning.
    Shared(Arc<VortexError>),
    /// A wrapper for errors from the Arrow library.
    ArrowError(arrow_schema::ArrowError, Box<Backtrace>),
    /// A wrapper for errors from the FlatBuffers library.
    #[cfg(feature = "flatbuffers")]
    FlatBuffersError(flatbuffers::InvalidFlatbuffer, Box<Backtrace>),
    /// A wrapper for formatting errors.
    FmtError(fmt::Error, Box<Backtrace>),
    /// A wrapper for IO errors.
    IOError(io::Error, Box<Backtrace>),
    /// A wrapper for errors from the standard library when converting a slice to an array.
    TryFromSliceError(std::array::TryFromSliceError, Box<Backtrace>),
    /// A wrapper for errors from the Object Store library.
    #[cfg(feature = "object_store")]
    ObjectStore(object_store::Error, Box<Backtrace>),
    /// A wrapper for errors from the Jiff library.
    JiffError(jiff::Error, Box<Backtrace>),
    /// A wrapper for Tokio join error.
    #[cfg(feature = "tokio")]
    JoinError(tokio::task::JoinError, Box<Backtrace>),
    /// A wrapper for URL parsing errors.
    UrlError(url::ParseError, Box<Backtrace>),
    /// Wrap errors for fallible integer casting.
    TryFromInt(TryFromIntError, Box<Backtrace>),
    /// Wrap serde and serde json errors
    #[cfg(feature = "serde")]
    SerdeJsonError(serde_json::Error, Box<Backtrace>),
    /// Wrap prost encode error
    #[cfg(feature = "prost")]
    ProstEncodeError(prost::EncodeError, Box<Backtrace>),
    /// Wrap prost decode error
    #[cfg(feature = "prost")]
    ProstDecodeError(prost::DecodeError, Box<Backtrace>),
    /// Wrap prost unknown enum value
    #[cfg(feature = "prost")]
    ProstUnknownEnumValue(prost::UnknownEnumValue, Box<Backtrace>),
}

impl VortexError {
    /// Adds additional context to an error.
    pub fn with_context<T: Into<ErrString>>(self, msg: T) -> Self {
        VortexError::Context(msg.into(), Box::new(self))
    }

    /// Wrap an a generic error into a Vortex error
    pub fn generic(err: Box<dyn Error + Send + Sync + 'static>) -> Self {
        Self::Generic(err, Box::new(Backtrace::capture()))
    }
}

impl Display for VortexError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            VortexError::Generic(err, backtrace) => {
                write!(f, "{err}\nBacktrace:\n{backtrace}")
            }
            VortexError::OutOfBounds(idx, start, stop, backtrace) => {
                write!(
                    f,
                    "index {idx} out of bounds from {start} to {stop}\nBacktrace:\n{backtrace}",
                )
            }
            VortexError::ComputeError(msg, backtrace) => {
                write!(f, "{msg}\nBacktrace:\n{backtrace}")
            }
            VortexError::InvalidArgument(msg, backtrace) => {
                write!(f, "{msg}\nBacktrace:\n{backtrace}")
            }
            VortexError::InvalidState(msg, backtrace) => {
                write!(f, "{msg}\nBacktrace:\n{backtrace}")
            }
            VortexError::InvalidSerde(msg, backtrace) => {
                write!(f, "{msg}\nBacktrace:\n{backtrace}")
            }
            VortexError::NotImplemented(func, by_whom, backtrace) => {
                write!(
                    f,
                    "function {func} not implemented for {by_whom}\nBacktrace:\n{backtrace}",
                )
            }
            VortexError::MismatchedTypes(expected, actual, backtrace) => {
                write!(
                    f,
                    "expected type: {expected} but instead got {actual}\nBacktrace:\n{backtrace}",
                )
            }
            VortexError::AssertionFailed(msg, backtrace) => {
                write!(f, "{msg}\nBacktrace:\n{backtrace}")
            }
            VortexError::Context(msg, inner) => {
                write!(f, "{msg}:\n  {inner}")
            }
            VortexError::Shared(inner) => Display::fmt(inner, f),
            VortexError::ArrowError(err, backtrace) => {
                write!(f, "{err}\nBacktrace:\n{backtrace}")
            }
            #[cfg(feature = "flatbuffers")]
            VortexError::FlatBuffersError(err, backtrace) => {
                write!(f, "{err}\nBacktrace:\n{backtrace}")
            }
            VortexError::FmtError(err, backtrace) => {
                write!(f, "{err}\nBacktrace:\n{backtrace}")
            }
            VortexError::IOError(err, backtrace) => {
                write!(f, "{err}\nBacktrace:\n{backtrace}")
            }
            VortexError::TryFromSliceError(err, backtrace) => {
                write!(f, "{err}\nBacktrace:\n{backtrace}")
            }
            #[cfg(feature = "object_store")]
            VortexError::ObjectStore(err, backtrace) => {
                write!(f, "{err}\nBacktrace:\n{backtrace}")
            }
            VortexError::JiffError(err, backtrace) => {
                write!(f, "{err}\nBacktrace:\n{backtrace}")
            }
            #[cfg(feature = "tokio")]
            VortexError::JoinError(err, backtrace) => {
                write!(f, "{err}\nBacktrace:\n{backtrace}")
            }
            VortexError::UrlError(err, backtrace) => {
                write!(f, "{err}\nBacktrace:\n{backtrace}")
            }
            VortexError::TryFromInt(err, backtrace) => {
                write!(f, "{err}\nBacktrace:\n{backtrace}")
            }
            #[cfg(feature = "serde")]
            VortexError::SerdeJsonError(err, backtrace) => {
                write!(f, "{err}\nBacktrace:\n{backtrace}")
            }
            #[cfg(feature = "prost")]
            VortexError::ProstEncodeError(err, backtrace) => {
                write!(f, "{err}\nBacktrace:\n{backtrace}")
            }
            #[cfg(feature = "prost")]
            VortexError::ProstDecodeError(err, backtrace) => {
                write!(f, "{err}\nBacktrace:\n{backtrace}")
            }
            #[cfg(feature = "prost")]
            VortexError::ProstUnknownEnumValue(err, backtrace) => {
                write!(f, "{err}\nBacktrace:\n{backtrace}")
            }
        }
    }
}

impl Error for VortexError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            VortexError::Generic(err, _) => Some(err.as_ref()),
            VortexError::Context(_, inner) => inner.source(),
            VortexError::Shared(inner) => inner.source(),
            VortexError::ArrowError(err, _) => Some(err),
            #[cfg(feature = "flatbuffers")]
            VortexError::FlatBuffersError(err, _) => Some(err),
            VortexError::IOError(err, _) => Some(err),
            #[cfg(feature = "object_store")]
            VortexError::ObjectStore(err, _) => Some(err),
            VortexError::JiffError(err, _) => Some(err),
            #[cfg(feature = "tokio")]
            VortexError::JoinError(err, _) => Some(err),
            VortexError::UrlError(err, _) => Some(err),
            #[cfg(feature = "serde")]
            VortexError::SerdeJsonError(err, _) => Some(err),
            #[cfg(feature = "prost")]
            VortexError::ProstEncodeError(err, _) => Some(err),
            #[cfg(feature = "prost")]
            VortexError::ProstDecodeError(err, _) => Some(err),
            #[cfg(feature = "prost")]
            VortexError::ProstUnknownEnumValue(err, _) => Some(err),
            _ => None,
        }
    }
}

impl Debug for VortexError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

/// A type alias for Results that return VortexErrors as their error type.
pub type VortexResult<T> = Result<T, VortexError>;

/// A vortex result that can be shared or cloned.
pub type SharedVortexResult<T> = Result<T, Arc<VortexError>>;

impl From<Arc<VortexError>> for VortexError {
    fn from(value: Arc<VortexError>) -> Self {
        Self::from(&value)
    }
}

impl From<&Arc<VortexError>> for VortexError {
    fn from(e: &Arc<VortexError>) -> Self {
        if let VortexError::Shared(e_inner) = e.as_ref() {
            // don't re-wrap
            VortexError::Shared(Arc::clone(e_inner))
        } else {
            VortexError::Shared(Arc::clone(e))
        }
    }
}

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
    ///
    /// # `&'static` message lifetime
    ///
    /// The panic string argument should be a string literal, hence the `&'static` lifetime. If
    /// you'd like to panic with a dynamic format string, consider using `unwrap_or_else` combined
    /// with the `vortex_panic!` macro instead.
    fn vortex_expect(self, msg: &'static str) -> Self::Output;
}

impl<T, E> VortexExpect for Result<T, E>
where
    E: Into<VortexError>,
{
    type Output = T;

    #[inline(always)]
    fn vortex_expect(self, msg: &'static str) -> Self::Output {
        self.map_err(|err| err.into())
            .unwrap_or_else(|e| vortex_panic!(e.with_context(msg.to_string())))
    }
}

impl<T> VortexExpect for Option<T> {
    type Output = T;

    #[inline(always)]
    fn vortex_expect(self, msg: &'static str) -> Self::Output {
        self.unwrap_or_else(|| {
            let err = VortexError::AssertionFailed(
                msg.to_string().into(),
                Box::new(Backtrace::capture()),
            );
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
            $crate::VortexError::AssertionFailed(err_string.into(), Box::new(Backtrace::capture()))
        )
    }};
    (IOError: $($tts:tt)*) => {{
        use std::backtrace::Backtrace;
        $crate::__private::must_use(
            $crate::VortexError::IOError(err_string.into(), Box::new(Backtrace::capture()))
        )
    }};
    (OutOfBounds: $idx:expr, $start:expr, $stop:expr) => {{
        use std::backtrace::Backtrace;
        $crate::__private::must_use(
            $crate::VortexError::OutOfBounds($idx, $start, $stop, Box::new(Backtrace::capture()))
        )
    }};
    (NotImplemented: $func:expr, $by_whom:expr) => {{
        use std::backtrace::Backtrace;
        $crate::__private::must_use(
            $crate::VortexError::NotImplemented($func.into(), format!("{}", $by_whom).into(), Box::new(Backtrace::capture()))
        )
    }};
    (MismatchedTypes: $expected:literal, $actual:expr) => {{
        use std::backtrace::Backtrace;
        $crate::__private::must_use(
            $crate::VortexError::MismatchedTypes($expected.into(), $actual.to_string().into(), Box::new(Backtrace::capture()))
        )
    }};
    (MismatchedTypes: $expected:expr, $actual:expr) => {{
        use std::backtrace::Backtrace;
        $crate::__private::must_use(
            $crate::VortexError::MismatchedTypes($expected.to_string().into(), $actual.to_string().into(), Box::new(Backtrace::capture()))
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
            $crate::VortexError::$variant(format!($fmt, $($arg),*).into(), Box::new(Backtrace::capture()))
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

/// A macro that mirrors `assert!` but instead of panicking on a failed condition,
/// it will immediately return an erroneous `VortexResult` to the calling context.
#[macro_export]
macro_rules! vortex_ensure {
    ($cond:expr) => {
        vortex_ensure!($cond, AssertionFailed: "{}", stringify!($cond));
    };
    ($cond:expr, $($tt:tt)*) => {
        if !$cond {
            $crate::vortex_bail!($($tt)*);
        }
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

impl From<arrow_schema::ArrowError> for VortexError {
    fn from(value: arrow_schema::ArrowError) -> Self {
        VortexError::ArrowError(value, Box::new(Backtrace::capture()))
    }
}

#[cfg(feature = "flatbuffers")]
impl From<flatbuffers::InvalidFlatbuffer> for VortexError {
    fn from(value: flatbuffers::InvalidFlatbuffer) -> Self {
        VortexError::FlatBuffersError(value, Box::new(Backtrace::capture()))
    }
}

impl From<io::Error> for VortexError {
    fn from(value: io::Error) -> Self {
        VortexError::IOError(value, Box::new(Backtrace::capture()))
    }
}

impl From<std::array::TryFromSliceError> for VortexError {
    fn from(value: std::array::TryFromSliceError) -> Self {
        VortexError::TryFromSliceError(value, Box::new(Backtrace::capture()))
    }
}

#[cfg(feature = "object_store")]
impl From<object_store::Error> for VortexError {
    fn from(value: object_store::Error) -> Self {
        VortexError::ObjectStore(value, Box::new(Backtrace::capture()))
    }
}

impl From<jiff::Error> for VortexError {
    fn from(value: jiff::Error) -> Self {
        VortexError::JiffError(value, Box::new(Backtrace::capture()))
    }
}

#[cfg(feature = "tokio")]
impl From<tokio::task::JoinError> for VortexError {
    fn from(value: tokio::task::JoinError) -> Self {
        if value.is_panic() {
            std::panic::resume_unwind(value.into_panic())
        } else {
            VortexError::JoinError(value, Box::new(Backtrace::capture()))
        }
    }
}

impl From<url::ParseError> for VortexError {
    fn from(value: url::ParseError) -> Self {
        VortexError::UrlError(value, Box::new(Backtrace::capture()))
    }
}

impl From<TryFromIntError> for VortexError {
    fn from(value: TryFromIntError) -> Self {
        VortexError::TryFromInt(value, Box::new(Backtrace::capture()))
    }
}

#[cfg(feature = "serde")]
impl From<serde_json::Error> for VortexError {
    fn from(value: serde_json::Error) -> Self {
        VortexError::SerdeJsonError(value, Box::new(Backtrace::capture()))
    }
}

#[cfg(feature = "prost")]
impl From<prost::EncodeError> for VortexError {
    fn from(value: prost::EncodeError) -> Self {
        VortexError::ProstEncodeError(value, Box::new(Backtrace::capture()))
    }
}

#[cfg(feature = "prost")]
impl From<prost::DecodeError> for VortexError {
    fn from(value: prost::DecodeError) -> Self {
        VortexError::ProstDecodeError(value, Box::new(Backtrace::capture()))
    }
}

#[cfg(feature = "prost")]
impl From<prost::UnknownEnumValue> for VortexError {
    fn from(value: prost::UnknownEnumValue) -> Self {
        VortexError::ProstUnknownEnumValue(value, Box::new(Backtrace::capture()))
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

impl<T> From<PoisonError<T>> for VortexError {
    fn from(_value: PoisonError<T>) -> Self {
        // We don't include the value since it may be sensitive.
        Self::InvalidState("Lock poisoned".into(), Box::new(Backtrace::capture()))
    }
}
