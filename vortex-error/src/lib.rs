// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]

//! This crate defines error & result types for Vortex.
//! It also contains a variety of useful macros for error handling.
//!
//! A [`VortexError`] is a human-readable message, not one enum variant per library that Vortex
//! happens to depend on. An error from a dependency is attached as the [`Error::source`] of a
//! [`VortexError`] and is recovered by downcasting that source.

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

/// The top-level error type for Vortex.
///
/// An error is a message, an optional underlying error, and a backtrace captured at
/// construction. Cloning is cheap: the payload is shared, never copied.
///
/// The payload sits behind one pointer, so a `VortexResult<T>` costs the same as `T` plus at
/// most a word on the success path. Most results fit in two registers and are returned in them,
/// rather than through a stack slot the caller must reserve and read back. Constructing an error
/// is no more expensive for it: the payload's allocation is the one the backtrace needed anyway.
#[derive(Clone)]
pub struct VortexError(Arc<ErrorInner>);

const _: () = assert!(size_of::<VortexError>() == size_of::<usize>());
const _: () = assert!(size_of::<VortexResult<()>>() == size_of::<usize>());

struct ErrorInner {
    /// `None` defers the message to the source, keeping the `?` conversion path allocation-free.
    message: Option<ErrString>,
    origin: Origin,
}

/// Where an error's source and backtrace come from.
enum Origin {
    /// Captured when this error was created.
    Own {
        source: Option<Box<dyn Error + Send + Sync + 'static>>,
        backtrace: Backtrace,
    },
    /// Shared with the error this one adds context to.
    ///
    /// Context is normally added in place. This is only for an error that has been cloned, whose
    /// payload the other clones still see.
    Context(VortexError),
}

impl VortexError {
    /// Creates an error carrying `message`.
    pub fn new<T: Into<ErrString>>(message: T) -> Self {
        Self::from_parts(Some(message.into()), None)
    }

    /// Wraps an underlying error as a Vortex error.
    ///
    /// The wrapped error is preserved as the [`Error::source`], so callers that care about the
    /// concrete type can recover it with [`Error::source`] and `downcast_ref`.
    pub fn external<E>(source: E) -> Self
    where
        E: Into<Box<dyn Error + Send + Sync + 'static>>,
    {
        Self::from_parts(None, Some(source.into()))
    }

    fn from_parts(
        message: Option<ErrString>,
        source: Option<Box<dyn Error + Send + Sync + 'static>>,
    ) -> Self {
        Self(Arc::new(ErrorInner {
            message,
            origin: Origin::Own {
                source,
                backtrace: Backtrace::capture(),
            },
        }))
    }

    /// Adds additional context to an error, preserving its source and backtrace.
    pub fn with_context<T: Into<ErrString>>(mut self, msg: T) -> Self {
        let msg: ErrString = msg.into();
        // Build the combined string directly: `msg` has already been through the
        // `VORTEX_PANIC_ON_ERR` check and must not trip it a second time.
        let message = ErrString(Cow::Owned(format!("{msg}:\n  {}", self.message_body())));
        if let Some(inner) = Arc::get_mut(&mut self.0) {
            inner.message = Some(message);
            return self;
        }
        Self(Arc::new(ErrorInner {
            message: Some(message),
            origin: Origin::Context(self),
        }))
    }

    /// The error message, falling back to the underlying error when none was provided.
    fn message_body(&self) -> Cow<'_, str> {
        match (&self.0.message, self.source_ref()) {
            (Some(message), _) => Cow::Borrowed(message.as_ref()),
            (None, Some(source)) => Cow::Owned(source.to_string()),
            (None, None) => Cow::Borrowed(""),
        }
    }

    fn source_ref(&self) -> Option<&(dyn Error + Send + Sync + 'static)> {
        match &self.0.origin {
            Origin::Own { source, .. } => source.as_deref(),
            Origin::Context(error) => error.source_ref(),
        }
    }

    fn backtrace(&self) -> &Backtrace {
        match &self.0.origin {
            Origin::Own { backtrace, .. } => backtrace,
            Origin::Context(error) => error.backtrace(),
        }
    }
}

impl Display for VortexError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match (&self.0.message, self.source_ref()) {
            (Some(message), _) => Display::fmt(message, f)?,
            (None, Some(source)) => Display::fmt(source, f)?,
            (None, None) => {}
        }
        let backtrace = self.backtrace();
        if backtrace.status() == BacktraceStatus::Captured {
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
        self.source_ref()
            .map(|source| source as &(dyn Error + 'static))
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
/// ```
/// use vortex_error::vortex_err;
///
/// let err = vortex_err!("negative length {}", -1);
/// assert!(err.to_string().starts_with("negative length -1"));
/// ```
#[macro_export]
macro_rules! vortex_err {
    (Context: $msg:literal, $err:expr) => {
        $crate::__private::context($msg, $err)
    };
    (External: $err:expr $(,)?) => {
        $crate::__private::external($err)
    };
    ($fmt:literal $(, $arg:expr)* $(,)?) => {
        $crate::__private::fmt_err(&format_args!($fmt $(, $arg)*))
    };
    ($err:expr $(,)?) => {
        $crate::__private::fmt_err(&format_args!("{}", $err))
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
/// Without a message the error names the condition.
#[macro_export]
macro_rules! vortex_ensure {
    ($cond:expr) => {
        $crate::vortex_ensure!($cond, "{}", stringify!($cond));
    };
    ($cond:expr, $($tt:tt)*) => {
        if !$cond {
            $crate::vortex_bail!($($tt)*);
        }
    };
}

/// A macro that mirrors `assert_eq!` but instead of panicking when `left != right`, it will
/// immediately return an erroneous `VortexResult` to the calling context.
///
/// Both values must implement [`PartialEq`] and [`Display`], and each is
/// evaluated exactly once. Use [`vortex_ensure!`] for values that do not implement `Display`.
///
/// By default the error shows both expressions and their values, so most callers do not need a
/// message:
///
/// ```
/// # use vortex_error::{VortexResult, vortex_ensure_eq};
/// fn check(len: usize) -> VortexResult<()> {
///     vortex_ensure_eq!(len, 2);
///     Ok(())
/// }
///
/// let err = check(3).unwrap_err().to_string();
/// assert!(err.starts_with("`len == 2`\n  left: 3\n right: 2"));
/// ```
///
/// A message replaces the expressions. The values are still appended, so the message does not
/// need to repeat them:
///
/// ```
/// # use vortex_error::{VortexResult, vortex_ensure_eq};
/// fn check(len: usize) -> VortexResult<()> {
///     vortex_ensure_eq!(len, 2, "map entries must have {} fields", 2);
///     Ok(())
/// }
///
/// let err = check(3).unwrap_err().to_string();
/// assert!(err.starts_with("map entries must have 2 fields\n  left: 3\n right: 2"));
/// ```
#[macro_export]
macro_rules! vortex_ensure_eq {
    ($left:expr, $right:expr $(,)?) => {
        match (&$left, &$right) {
            (left_val, right_val) => {
                if *left_val != *right_val {
                    $crate::vortex_bail!(
                        "`{} == {}`\n  left: {}\n right: {}",
                        stringify!($left),
                        stringify!($right),
                        left_val,
                        right_val
                    );
                }
            }
        }
    };
    ($left:expr, $right:expr, $fmt:literal $(, $arg:expr)* $(,)?) => {
        match (&$left, &$right) {
            (left_val, right_val) => {
                if *left_val != *right_val {
                    $crate::vortex_bail!(
                        "{}\n  left: {}\n right: {}",
                        format_args!($fmt $(, $arg)*),
                        left_val,
                        right_val
                    );
                }
            }
        }
    };
}

/// A convenient macro for panicking with a VortexError in the presence of a programmer error
/// (e.g., an invariant has been violated).
///
/// It takes a message like [`vortex_err!`], or an existing error with an optional message to add
/// as context.
#[macro_export]
macro_rules! vortex_panic {
    (Context: $msg:literal, $err:expr) => {{
        $crate::vortex_panic!($crate::vortex_err!(Context: $msg, $err))
    }};
    // Before the `$err:expr` arms, which a bare message would otherwise match.
    ($fmt:literal $(, $arg:expr)* $(,)?) => {
        $crate::vortex_panic!($crate::vortex_err!($fmt $(, $arg)*))
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
        VortexError::external(value)
    }
}

#[cfg(feature = "flatbuffers")]
impl From<flatbuffers::InvalidFlatbuffer> for VortexError {
    fn from(value: flatbuffers::InvalidFlatbuffer) -> Self {
        VortexError::external(value)
    }
}

impl From<io::Error> for VortexError {
    fn from(value: io::Error) -> Self {
        VortexError::external(value)
    }
}

#[cfg(feature = "object_store")]
impl From<object_store::Error> for VortexError {
    fn from(value: object_store::Error) -> Self {
        VortexError::external(value)
    }
}

impl From<jiff::Error> for VortexError {
    fn from(value: jiff::Error) -> Self {
        VortexError::external(value)
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
        VortexError::external(value)
    }
}

impl From<prost::EncodeError> for VortexError {
    fn from(value: prost::EncodeError) -> Self {
        VortexError::external(value)
    }
}

impl From<prost::DecodeError> for VortexError {
    fn from(value: prost::DecodeError) -> Self {
        VortexError::external(value)
    }
}

impl From<prost::UnknownEnumValue> for VortexError {
    fn from(value: prost::UnknownEnumValue) -> Self {
        VortexError::external(value)
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
    pub fn fmt_err(args: &Arguments<'_>) -> VortexError {
        VortexError::new(args.to_string())
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
        panic!("{}", VortexError::new(msg))
    }
}
