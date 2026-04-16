// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]

//! This crate defines error & result types for Vortex.
//! It also contains a variety of useful macros for error handling.

use std::backtrace::Backtrace;
use std::backtrace::BacktraceStatus;
use std::borrow::Cow;
use std::convert::Infallible;
use std::env;
use std::error::Error;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::io;
use std::num::TryFromIntError;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::LazyLock;

/// A string that can be used as an error message.
#[derive(Debug)]
pub struct ErrString(Cow<'static, str>);

#[expect(
    clippy::fallible_impl_from,
    reason = "intentionally panic in debug mode when VORTEX_PANIC_ON_ERR is set"
)]
impl<T> From<T> for ErrString
where
    T: Into<Cow<'static, str>>,
{
    #[expect(
        clippy::panic,
        reason = "intentionally panic in debug mode when VORTEX_PANIC_ON_ERR is set"
    )]
    fn from(msg: T) -> Self {
        if panic_on_err() {
            panic!("{}\nBacktrace:\n{}", msg.into(), Backtrace::capture());
        } else {
            Self(msg.into())
        }
    }
}

fn panic_on_err() -> bool {
    static PANIC_ON_ERR: LazyLock<bool> =
        LazyLock::new(|| env::var("VORTEX_PANIC_ON_ERR").is_ok_and(|v| v == "1"));
    *PANIC_ON_ERR
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
    /// A catch-all error variant
    Other(ErrString, Box<Backtrace>),
    /// A wrapped external error
    External(Box<dyn Error + Send + Sync + 'static>, Box<Backtrace>),
    /// An index is out of bounds.
    OutOfBounds(usize, usize, usize, Box<Backtrace>),
    /// An error occurred while executing a compute kernel.
    Compute(ErrString, Box<Backtrace>),
    /// An invalid argument was provided.
    InvalidArgument(ErrString, Box<Backtrace>),
    /// An error occurred while serializing or deserializing.
    Serde(ErrString, Box<Backtrace>),
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
    Arrow(arrow_schema::ArrowError, Box<Backtrace>),
    /// A wrapper for errors from the FlatBuffers library.
    #[cfg(feature = "flatbuffers")]
    FlatBuffers(flatbuffers::InvalidFlatbuffer, Box<Backtrace>),
    /// A wrapper for formatting errors.
    Fmt(fmt::Error, Box<Backtrace>),
    /// A wrapper for IO errors.
    Io(io::Error, Box<Backtrace>),
    /// A wrapper for errors from the Object Store library.
    #[cfg(feature = "object_store")]
    ObjectStore(object_store::Error, Box<Backtrace>),
    /// A wrapper for errors from the Jiff library.
    Jiff(jiff::Error, Box<Backtrace>),
    /// A wrapper for Tokio join error.
    #[cfg(feature = "tokio")]
    Join(tokio::task::JoinError, Box<Backtrace>),
    /// Wrap errors for fallible integer casting.
    TryFromInt(TryFromIntError, Box<Backtrace>),
    /// Wrap protobuf-related errors
    Prost(Box<dyn Error + Send + Sync + 'static>, Box<Backtrace>),
}

impl VortexError {
    /// Adds additional context to an error.
    pub fn with_context<T: Into<ErrString>>(self, msg: T) -> Self {
        VortexError::Context(msg.into(), Box::new(self))
    }

    /// Error prefix by variant
    fn variant_prefix(&self) -> &'static str {
        use VortexError::*;

        match self {
            Other(..) => "Other error: ",
            External(..) => "External error: ",
            OutOfBounds(..) => "Out of bounds error: ",
            Compute(..) => "Compute error: ",
            InvalidArgument(..) => "Invalid argument error: ",
            Serde(..) => "Serde error: ",
            NotImplemented(..) => "Not implemented error: ",
            MismatchedTypes(..) => "Mismatched types error: ",
            AssertionFailed(..) => "Assertion failed error: ",
            Context(..) | Shared(..) => "", // basically delegate to the underlying one
            Arrow(..) => "Arrow error: ",
            #[cfg(feature = "flatbuffers")]
            FlatBuffers(..) => "Flat buffers error: ",
            Fmt(..) => "Fmt: ",
            Io(..) => "Io: ",
            #[cfg(feature = "object_store")]
            ObjectStore(..) => "Object store error: ",
            Jiff(..) => "Jiff error: ",
            #[cfg(feature = "tokio")]
            Join(..) => "Tokio join error:",
            TryFromInt(..) => "Try from int error:",
            Prost(..) => "Prost error:",
        }
    }

    fn backtrace(&self) -> Option<&Backtrace> {
        use VortexError::*;

        match self {
            Other(.., bt) => Some(bt.as_ref()),
            External(.., bt) => Some(bt.as_ref()),
            OutOfBounds(.., bt) => Some(bt.as_ref()),
            Compute(.., bt) => Some(bt.as_ref()),
            InvalidArgument(.., bt) => Some(bt.as_ref()),
            Serde(.., bt) => Some(bt.as_ref()),
            NotImplemented(.., bt) => Some(bt.as_ref()),
            MismatchedTypes(.., bt) => Some(bt.as_ref()),
            AssertionFailed(.., bt) => Some(bt.as_ref()),
            Arrow(.., bt) => Some(bt.as_ref()),
            #[cfg(feature = "flatbuffers")]
            FlatBuffers(.., bt) => Some(bt.as_ref()),
            Fmt(.., bt) => Some(bt.as_ref()),
            Io(.., bt) => Some(bt.as_ref()),
            #[cfg(feature = "object_store")]
            ObjectStore(.., bt) => Some(bt.as_ref()),
            Jiff(.., bt) => Some(bt.as_ref()),
            #[cfg(feature = "tokio")]
            Join(.., bt) => Some(bt.as_ref()),
            TryFromInt(.., bt) => Some(bt.as_ref()),
            Prost(.., bt) => Some(bt.as_ref()),
            Context(_, inner) => inner.backtrace(),
            Shared(inner) => inner.backtrace(),
        }
    }

    fn message(&self) -> String {
        use VortexError::*;

        match self {
            Other(msg, _) => msg.to_string(),
            External(err, _) => err.to_string(),
            OutOfBounds(idx, start, stop, _) => {
                format!("index {idx} out of bounds from {start} to {stop}")
            }
            Compute(msg, _) | InvalidArgument(msg, _) | Serde(msg, _) | AssertionFailed(msg, _) => {
                format!("{msg}")
            }
            NotImplemented(func, by_whom, _) => {
                format!("function {func} not implemented for {by_whom}")
            }
            MismatchedTypes(expected, actual, _) => {
                format!("expected type: {expected} but instead got {actual}")
            }
            Context(msg, inner) => {
                format!("{msg}:\n  {inner}")
            }
            Shared(inner) => inner.message(),
            Arrow(err, _) => {
                format!("{err}")
            }
            #[cfg(feature = "flatbuffers")]
            FlatBuffers(err, _) => {
                format!("{err}")
            }
            Fmt(err, _) => {
                format!("{err}")
            }
            Io(err, _) => {
                format!("{err}")
            }
            #[cfg(feature = "object_store")]
            ObjectStore(err, _) => {
                format!("{err}")
            }
            Jiff(err, _) => {
                format!("{err}")
            }
            #[cfg(feature = "tokio")]
            Join(err, _) => {
                format!("{err}")
            }
            TryFromInt(err, _) => {
                format!("{err}")
            }
            Prost(err, _) => {
                format!("{err}")
            }
        }
    }
}

impl Display for VortexError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.variant_prefix())?;
        write!(f, "{}", self.message())?;
        if let Some(backtrace) = self.backtrace()
            && backtrace.status() == BacktraceStatus::Captured
        {
            write!(f, "\nBacktrace:\n{backtrace}")?;
        }

        Ok(())
    }
}

impl Debug for VortexError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl Error for VortexError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        use VortexError::*;

        match self {
            External(err, _) => Some(err.as_ref()),
            Context(_, inner) => inner.source(),
            Shared(inner) => inner.source(),
            Arrow(err, _) => Some(err),
            #[cfg(feature = "flatbuffers")]
            FlatBuffers(err, _) => Some(err),
            Io(err, _) => Some(err),
            #[cfg(feature = "object_store")]
            ObjectStore(err, _) => Some(err),
            Jiff(err, _) => Some(err),
            #[cfg(feature = "tokio")]
            Join(err, _) => Some(err),
            Prost(err, _) => Some(err.as_ref()),
            _ => None,
        }
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
    (Other: $($tts:tt)*) => {{
        use std::backtrace::Backtrace;
        let err_string = format!($($tts)*);
        $crate::__private::must_use(
            $crate::VortexError::Other(err_string.into(), Box::new(Backtrace::capture()))
        )
    }};
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
    (External: $err:expr) => {{
        use std::backtrace::Backtrace;
        $crate::__private::must_use(
            $crate::VortexError::External($err.into(), Box::new(Backtrace::capture()))
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
        $crate::vortex_err!(Other: $fmt, $($arg),*)
    };
}

/// A convenience macro for returning a VortexError.
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

/// A macro that mirrors `assert_eq!` but instead of panicking when left != right,
/// it will immediately return an erroneous `VortexResult` to the calling context.
#[macro_export]
macro_rules! vortex_ensure_eq {
    ($left:expr, $right:expr) => {
        $crate::vortex_ensure_eq!($left, $right, AssertionFailed: "{} != {}: {:?} != {:?}", stringify!($left), stringify!($right), $left, $right);
    };
    ($left:expr, $right:expr, $($tt:tt)*) => {
        if $left != $right {
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
        VortexError::Arrow(value, Box::new(Backtrace::capture()))
    }
}

#[cfg(feature = "flatbuffers")]
impl From<flatbuffers::InvalidFlatbuffer> for VortexError {
    fn from(value: flatbuffers::InvalidFlatbuffer) -> Self {
        VortexError::FlatBuffers(value, Box::new(Backtrace::capture()))
    }
}

impl From<io::Error> for VortexError {
    fn from(value: io::Error) -> Self {
        VortexError::Io(value, Box::new(Backtrace::capture()))
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
        VortexError::Jiff(value, Box::new(Backtrace::capture()))
    }
}

#[cfg(feature = "tokio")]
impl From<tokio::task::JoinError> for VortexError {
    fn from(value: tokio::task::JoinError) -> Self {
        if value.is_panic() {
            std::panic::resume_unwind(value.into_panic())
        } else {
            VortexError::Join(value, Box::new(Backtrace::capture()))
        }
    }
}

impl From<TryFromIntError> for VortexError {
    fn from(value: TryFromIntError) -> Self {
        VortexError::TryFromInt(value, Box::new(Backtrace::capture()))
    }
}

impl From<prost::EncodeError> for VortexError {
    fn from(value: prost::EncodeError) -> Self {
        Self::Prost(Box::new(value), Box::new(Backtrace::capture()))
    }
}

impl From<prost::DecodeError> for VortexError {
    fn from(value: prost::DecodeError) -> Self {
        Self::Prost(Box::new(value), Box::new(Backtrace::capture()))
    }
}

impl From<prost::UnknownEnumValue> for VortexError {
    fn from(value: prost::UnknownEnumValue) -> Self {
        Self::Prost(Box::new(value), Box::new(Backtrace::capture()))
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
