// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// A macro for constructing bit-buffers akin to `vec![..]`.
///
/// Supports multiple syntaxes:
/// - `bitbuffer![]` - empty buffer
/// - `bitbuffer![value; count]` - fill with value
/// - `bitbuffer![expr, expr, ...]` - comma-separated boolean expressions
/// - `bitbuffer![0 1 0 1]` - space-separated bit literals (0s and 1s)
#[macro_export]
macro_rules! bitbuffer {
    // Internal rule to convert a single bit (0 or 1) to bool
    (@bit 0) => { false };
    (@bit 1) => { true };

    () => (
        $crate::BitBuffer::empty()
    );

    // We capture single-element 0/1 cases to avoid ambiguity with the
    // comma-separated expression case.
    (0) => {
        $crate::BitBuffer::from_iter([false])
    };
    (1) => {
        $crate::BitBuffer::from_iter([true])
    };

    ($elem:expr; $n:expr) => (
        $crate::BitBuffer::full($elem, $n)
    );
    ($($x:expr),+ $(,)?) => (
        $crate::BitBuffer::from_iter([$($x),+])
    );
    ($($bit:tt)+) => {
        $crate::BitBuffer::from_iter([$( $crate::bitbuffer!(@bit $bit) ),+])
    };
}

/// A macro for constructing bit-buffers akin to `vec![..]`.
///
/// Supports multiple syntaxes:
/// - `bitbuffer_mut![]` - empty buffer
/// - `bitbuffer_mut![value; count]` - fill with value
/// - `bitbuffer_mut![expr, expr, ...]` - comma-separated boolean expressions
/// - `bitbuffer_mut![0 1 0 1]` - space-separated bit literals (0s and 1s)
#[macro_export]
macro_rules! bitbuffer_mut {
    // Internal rule to convert a single bit (0 or 1) to bool
    (@bit 0) => { false };
    (@bit 1) => { true };

    () => (
        $crate::BitBufferMut::empty()
    );

    // We capture single-element 0/1 cases to avoid ambiguity with the
    // comma-separated expression case.
    (0) => {
        $crate::BitBuffer::from_iter([false])
    };
    (1) => {
        $crate::BitBuffer::from_iter([true])
    };

    ($elem:expr; $n:expr) => (
        $crate::BitBufferMut::full($elem, $n)
    );
    ($($x:expr),+ $(,)?) => (
        $crate::BitBufferMut::from_iter([$($x),+])
    );
    ($($bit:tt)+) => {
        $crate::BitBufferMut::from_iter([$( $crate::bitbuffer_mut!(@bit $bit) ),+])
    };
}
