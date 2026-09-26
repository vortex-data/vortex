// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]

//! This crate defines error & result types for Vortex.
//! It also contains a variety of useful macros for error handling.
//!
//! Vortex models errors the way Python models exceptions: a small, stable set of classes
//! ([`VortexErrorKind`]) carrying a human-readable message, rather than one enum variant per
//! library that Vortex happens to depend on. An error from a dependency is attached as the
//! [`Error::source`] of a [`VortexError`] and is recovered by downcasting that source.

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
#[derive(Debug, Clone)]
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

/// The classification of a [`VortexError`], analogous to a Python exception class.
///
/// The set is deliberately small and is mirrored by the `vx_error_code` enum in the C API, so a
/// kind only earns its place if a caller would branch on it. Failures originating in a dependency
/// are classified by what they mean to Vortex, not by which crate produced them.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum VortexErrorKind {
    /// An otherwise unclassified error. The analogue of Python's `RuntimeError`.
    Other,
    /// An index is out of bounds. The analogue of Python's `IndexError`.
    OutOfBounds,
    /// A name was looked up and nothing was bound to it. The analogue of Python's `KeyError`.
    NotFound,
    /// A numeric value does not fit its target type. The analogue of Python's `OverflowError`.
    Overflow,
    /// An invalid argument was provided. The analogue of Python's `ValueError`.
    InvalidArgument,
    /// An error occurred while serializing or deserializing. Closest to Python's `ValueError`,
    /// as raised by `json.JSONDecodeError`.
    Serde,
    /// An unimplemented function was called. The analogue of Python's `NotImplementedError`.
    NotImplemented,
    /// A value did not have the expected type. The analogue of Python's `TypeError`.
    MismatchedTypes,
    /// An internal invariant was violated. The analogue of Python's `AssertionError`.
    AssertionFailed,
    /// An IO operation, or a call into the operating system or a device driver, failed. The
    /// analogue of Python's `OSError`, which covers any system function that reports an error.
    Io,
}

impl VortexErrorKind {
    /// The human-readable prefix used when displaying an error of this kind.
    const fn prefix(self) -> &'static str {
        match self {
            Self::Other => "Other error: ",
            Self::OutOfBounds => "Out of bounds error: ",
            Self::NotFound => "Not found error: ",
            Self::Overflow => "Overflow error: ",
            Self::InvalidArgument => "Invalid argument error: ",
            Self::Serde => "Serde error: ",
            Self::NotImplemented => "Not implemented error: ",
            Self::MismatchedTypes => "Mismatched types error: ",
            Self::AssertionFailed => "Assertion failed error: ",
            Self::Io => "IO error: ",
        }
    }
}

/// The top-level error type for Vortex.
///
/// An error is a [`VortexErrorKind`], a message, an optional underlying error, and a backtrace
/// captured at construction. Cloning is cheap: the payload is shared, never copied.
#[derive(Clone)]
pub struct VortexError {
    kind: VortexErrorKind,
    /// `None` defers the message to `source`, keeping the `?` conversion path allocation-free.
    message: Option<ErrString>,
    source: Option<Arc<dyn Error + Send + Sync + 'static>>,
    backtrace: Arc<Backtrace>,
}

const _: () = assert!(size_of::<VortexError>() <= 56);

impl VortexError {
    /// Creates an error of the given kind carrying `message`.
    pub fn new<T: Into<ErrString>>(kind: VortexErrorKind, message: T) -> Self {
        Self {
            kind,
            message: Some(message.into()),
            source: None,
            backtrace: Arc::new(Backtrace::capture()),
        }
    }

    /// Wraps an underlying error as a Vortex error of the given kind.
    ///
    /// The wrapped error is preserved as the [`Error::source`], so callers that care about the
    /// concrete type can recover it with [`Error::source`] and `downcast_ref`.
    pub fn wrap<E>(kind: VortexErrorKind, source: E) -> Self
    where
        E: Into<Box<dyn Error + Send + Sync + 'static>>,
    {
        Self {
            kind,
            message: None,
            source: Some(Arc::from(source.into())),
            backtrace: Arc::new(Backtrace::capture()),
        }
    }

    /// Wraps an underlying error that does not fit any more specific [`VortexErrorKind`].
    pub fn external<E>(source: E) -> Self
    where
        E: Into<Box<dyn Error + Send + Sync + 'static>>,
    {
        Self::wrap(VortexErrorKind::Other, source)
    }

    /// The classification of this error.
    pub fn kind(&self) -> VortexErrorKind {
        self.kind
    }

    /// Adds additional context to an error, preserving its kind, source and backtrace.
    pub fn with_context<T: Into<ErrString>>(mut self, msg: T) -> Self {
        let msg: ErrString = msg.into();
        // Build the combined string directly: `msg` has already been through the
        // `VORTEX_PANIC_ON_ERR` check and must not trip it a second time.
        self.message = Some(ErrString(Cow::Owned(format!(
            "{msg}:\n  {}",
            self.message_body()
        ))));
        self
    }

    /// The error message, falling back to the underlying error when none was provided.
    fn message_body(&self) -> Cow<'_, str> {
        match (&self.message, &self.source) {
            (Some(message), _) => Cow::Borrowed(message.as_ref()),
            (None, Some(source)) => Cow::Owned(source.to_string()),
            (None, None) => Cow::Borrowed(""),
        }
    }
}

impl Display for VortexError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.kind.prefix())?;
        match (&self.message, &self.source) {
            (Some(message), _) => Display::fmt(message, f)?,
            (None, Some(source)) => Display::fmt(source, f)?,
            (None, None) => {}
        }
        if self.backtrace.status() == BacktraceStatus::Captured {
            write!(f, "\nBacktrace:\n{}", self.backtrace)?;
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
        self.source
            .as_ref()
            .map(|source| source.as_ref() as &(dyn Error + 'static))
    }
}

/// A type alias for Results that return VortexErrors as their error type.
pub type VortexResult<T> = Result<T, VortexError>;

/// A vortex result that can be shared or cloned.
pub type SharedVortexResult<T> = Result<T, Arc<VortexError>>;

impl From<Arc<VortexError>> for VortexError {
    fn from(value: Arc<VortexError>) -> Self {
        Arc::try_unwrap(value).unwrap_or_else(|value| (*value).clone())
    }
}

impl From<&Arc<VortexError>> for VortexError {
    fn from(value: &Arc<VortexError>) -> Self {
        (**value).clone()
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

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn vortex_expect(self, msg: &'static str) -> Self::Output {
        match self {
            Ok(value) => value,
            Err(err) => __private::result_expect_failed(err, msg),
        }
    }
}

impl<T> VortexExpect for Option<T> {
    type Output = T;

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn vortex_expect(self, msg: &'static str) -> Self::Output {
        match self {
            Some(value) => value,
            None => __private::option_expect_failed(msg),
        }
    }
}

/// A convenient macro for creating a VortexError.
///
/// The leading `Kind:` names the [`VortexErrorKind`] and is required: pick the kind that says
/// what went wrong, and `Other` only when none does. Every kind takes the same
/// `Kind: "format", args..` shape, so no kind gets a bespoke argument grammar that a format string
/// could be mistaken for.
///
/// ```
/// use vortex_error::{VortexErrorKind, vortex_err};
///
/// let err = vortex_err!(InvalidArgument: "negative length {}", -1);
/// assert_eq!(err.kind(), VortexErrorKind::InvalidArgument);
/// ```
///
/// A message without a kind does not compile:
///
/// ```compile_fail
/// let err = vortex_error::vortex_err!("negative length {}", -1);
/// ```
#[macro_export]
macro_rules! vortex_err {
    (Context: $msg:literal, $err:expr) => {
        $crate::__private::context($msg, $err)
    };
    (External: $err:expr $(,)?) => {
        $crate::__private::external($err)
    };
    ($kind:ident: $fmt:literal $(, $arg:expr)* $(,)?) => {
        $crate::__private::fmt_err($crate::VortexErrorKind::$kind, &format_args!($fmt $(, $arg)*))
    };
    ($kind:ident: $err:expr $(,)?) => {
        $crate::__private::fmt_err($crate::VortexErrorKind::$kind, &format_args!("{}", $err))
    };
    ($fmt:literal $($rest:tt)*) => {
        ::core::compile_error!(::core::concat!(
            "vortex_err! needs a kind before its message, e.g. `vortex_err!(InvalidArgument: ",
            ::core::stringify!($fmt),
            ")`"
        ))
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
///
/// Without a message the error is an [`VortexErrorKind::AssertionFailed`] naming the condition. A
/// message needs a leading `Kind:`, as in [`vortex_err!`]:
///
/// ```compile_fail
/// fn check(len: usize) -> vortex_error::VortexResult<()> {
///     vortex_error::vortex_ensure!(len > 0, "length must be positive");
///     Ok(())
/// }
/// ```
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
///
/// Like [`vortex_err!`], a message needs a leading `Kind:`; an existing error can be passed as is.
///
/// ```compile_fail
/// vortex_error::vortex_panic!("unreachable state {}", 3);
/// ```
#[macro_export]
macro_rules! vortex_panic {
    (Context: $msg:literal, $err:expr) => {{
        $crate::vortex_panic!($crate::vortex_err!(Context: $msg, $err))
    }};
    ($kind:ident: $fmt:literal $(, $arg:expr)* $(,)?) => {
        $crate::vortex_panic!($crate::vortex_err!($kind: $fmt, $($arg),*))
    };
    // Before the `$err:expr` arms, which a bare message would otherwise match.
    ($fmt:literal $($rest:tt)*) => {
        ::core::compile_error!(::core::concat!(
            "vortex_panic! needs a kind before its message, e.g. `vortex_panic!(AssertionFailed: ",
            ::core::stringify!($fmt),
            ")`"
        ))
    };
    ($err:expr, $fmt:literal $(, $arg:expr)* $(,)?) => {
        $crate::__private::panic_err_ctx($err, &format_args!($fmt $(, $arg)*))
    };
    ($err:expr) => {
        $crate::__private::panic_err($err)
    };
}

impl From<Infallible> for VortexError {
    fn from(_: Infallible) -> Self {
        unreachable!()
    }
}

impl From<arrow_schema::ArrowError> for VortexError {
    fn from(value: arrow_schema::ArrowError) -> Self {
        use arrow_schema::ArrowError::*;
        let kind = match &value {
            NotYetImplemented(_) => VortexErrorKind::NotImplemented,
            // A cast or parse Arrow rejects is a value it cannot represent, not a type it
            // cannot handle, which is why `int("x")` is a ValueError.
            CastError(_) | ParseError(_) | DivideByZero | InvalidArgumentError(_) => {
                VortexErrorKind::InvalidArgument
            }
            SchemaError(_) => VortexErrorKind::MismatchedTypes,
            ArithmeticOverflow(_)
            | DictionaryKeyOverflowError
            | RunEndIndexOverflowError
            | OffsetOverflowError(_) => VortexErrorKind::Overflow,
            CsvError(_) | JsonError(_) | IpcError(_) | ParquetError(_) | CDataInterface(_) => {
                VortexErrorKind::Serde
            }
            IoError(..) => VortexErrorKind::Io,
            AvroError(_) => VortexErrorKind::Serde,
            ExternalError(_) | MemoryError(_) | ComputeError(_) => VortexErrorKind::Other,
        };
        VortexError::wrap(kind, value)
    }
}

#[cfg(feature = "flatbuffers")]
impl From<flatbuffers::InvalidFlatbuffer> for VortexError {
    fn from(value: flatbuffers::InvalidFlatbuffer) -> Self {
        VortexError::wrap(VortexErrorKind::Serde, value)
    }
}

impl From<io::Error> for VortexError {
    fn from(value: io::Error) -> Self {
        VortexError::wrap(VortexErrorKind::Io, value)
    }
}

#[cfg(feature = "object_store")]
impl From<object_store::Error> for VortexError {
    fn from(value: object_store::Error) -> Self {
        VortexError::wrap(VortexErrorKind::Io, value)
    }
}

impl From<jiff::Error> for VortexError {
    fn from(value: jiff::Error) -> Self {
        // A date that does not exist and a unit that makes no sense for the call are both what
        // Python's datetime raises ValueError for; jiff reports anything else only as text.
        let kind = if value.is_range() || value.is_invalid_parameter() {
            VortexErrorKind::InvalidArgument
        } else {
            VortexErrorKind::Other
        };
        VortexError::wrap(kind, value)
    }
}

#[cfg(feature = "tokio")]
impl From<tokio::task::JoinError> for VortexError {
    fn from(value: tokio::task::JoinError) -> Self {
        if value.is_panic() {
            std::panic::resume_unwind(value.into_panic())
        } else {
            VortexError::external(value)
        }
    }
}

impl From<TryFromIntError> for VortexError {
    fn from(value: TryFromIntError) -> Self {
        VortexError::wrap(VortexErrorKind::Overflow, value)
    }
}

impl From<prost::EncodeError> for VortexError {
    fn from(value: prost::EncodeError) -> Self {
        VortexError::wrap(VortexErrorKind::Serde, value)
    }
}

impl From<prost::DecodeError> for VortexError {
    fn from(value: prost::DecodeError) -> Self {
        VortexError::wrap(VortexErrorKind::Serde, value)
    }
}

impl From<prost::UnknownEnumValue> for VortexError {
    fn from(value: prost::UnknownEnumValue) -> Self {
        VortexError::wrap(VortexErrorKind::Serde, value)
    }
}

/// We want our error handlers to be fast in the happy path, so we inline them.
/// However, if we inline them as is, the call site gets the cold branch with
/// panic! macros which gets expanded into backtrace collection and throwing.
/// LLVM inliner/vectoriser in turn doesn't vectorize some code.
/// So we inline only the happy path, the error path gets to a cold handler which
/// is never inlined and thus doesn't prevent vectorization.
#[doc(hidden)]
#[expect(
    clippy::panic,
    reason = "cold handlers for vortex_panic!/vortex_expect intentionally panic"
)]
pub mod __private {

    use std::error::Error;
    use std::fmt::Arguments;

    use crate::VortexError;
    use crate::VortexErrorKind;

    #[doc(hidden)]
    #[inline]
    #[cold]
    #[must_use]
    pub const fn must_use(error: VortexError) -> VortexError {
        error
    }

    #[doc(hidden)]
    #[cold]
    #[inline(never)]
    #[must_use]
    pub fn fmt_err(kind: VortexErrorKind, args: &Arguments<'_>) -> VortexError {
        VortexError::new(kind, args.to_string())
    }

    #[doc(hidden)]
    #[cold]
    #[inline(never)]
    #[must_use]
    pub fn context(msg: &'static str, err: VortexError) -> VortexError {
        err.with_context(msg)
    }

    #[doc(hidden)]
    #[cold]
    #[inline(never)]
    #[must_use]
    pub fn external(err: impl Into<Box<dyn Error + Send + Sync + 'static>>) -> VortexError {
        VortexError::external(err)
    }

    #[doc(hidden)]
    #[cold]
    #[inline(never)]
    pub fn panic_err(err: VortexError) -> ! {
        panic!("{err}")
    }

    #[doc(hidden)]
    #[cold]
    #[inline(never)]
    pub fn panic_err_ctx(err: VortexError, args: &Arguments<'_>) -> ! {
        panic!("{}", err.with_context(args.to_string()))
    }

    #[doc(hidden)]
    #[cold]
    #[inline(never)]
    pub fn result_expect_failed(err: impl Into<VortexError>, msg: &'static str) -> ! {
        panic!("{}", err.into().with_context(msg))
    }

    #[doc(hidden)]
    #[cold]
    #[inline(never)]
    pub fn option_expect_failed(msg: &'static str) -> ! {
        panic!(
            "{}",
            VortexError::new(VortexErrorKind::AssertionFailed, msg)
        )
    }
}
