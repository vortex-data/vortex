// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// Dispatches a byte width to a compile-time `const W: usize`.
///
/// The default specialized widths are 1, 2, 4, 8, 16, and 32 bytes. Pass `widths = [...]` to
/// specialize a different set of nonzero powers of two. The input is a `usize` expression evaluated
/// once. Any other width, including zero, reaches the required fallback without rounding.
///
/// The body and fallback are ordinary match arms: they can move values, return, and propagate
/// errors from the enclosing function. Both must produce the same result type.
///
/// # Examples
///
/// ```
/// use vortex_buffer::match_each_power_of_two_width;
///
/// let width = match_each_power_of_two_width!(
///     8,
///     |W| { Ok::<_, usize>(size_of::<[u8; W]>()) },
///     other => { Err(other) },
/// );
/// assert_eq!(width, Ok(8));
///
/// let width = match_each_power_of_two_width!(
///     64,
///     widths = [16, 64],
///     |W| { W },
///     other => { other },
/// );
/// assert_eq!(width, 64);
/// ```
///
/// Specializations must be nonzero powers of two:
///
/// ```compile_fail
/// use vortex_buffer::match_each_power_of_two_width;
///
/// match_each_power_of_two_width!(3, widths = [3], |W| { W }, other => { other });
/// ```
#[macro_export]
macro_rules! match_each_power_of_two_width {
    (
        $byte_width:expr,
        widths = [$($width:literal),+ $(,)?],
        |$W:ident| $body:block,
        $other:pat => $fallback:block $(,)?
    ) => {{
        let byte_width: usize = $byte_width;
        match byte_width {
            $(
                $width => {
                    const $W: usize = $width;
                    const { assert!($W.is_power_of_two(), "width must be a nonzero power of two") };
                    $body
                },
            )+
            $other => $fallback,
        }
    }};
    (
        $byte_width:expr,
        |$W:ident| $body:block,
        $other:pat => $fallback:block $(,)?
    ) => {
        $crate::match_each_power_of_two_width!(
            $byte_width,
            widths = [1, 2, 4, 8, 16, 32],
            |$W| $body,
            $other => $fallback,
        )
    };
}

/// A macro for constructing buffers akin to `vec![..]`.
#[macro_export]
macro_rules! buffer {
    () => (
        $crate::Buffer::empty()
    );
    ($start:tt .. $end:tt) => (
        $crate::Buffer::from_iter($start..$end)
    );
    ($start:tt ..= $end:tt) => (
        $crate::Buffer::from_iter($start..=$end)
    );
    ($elem:expr; $n:expr) => (
        $crate::Buffer::full($elem, $n)
    );
    ($($x:expr),+ $(,)?) => (
        $crate::Buffer::from_iter([$($x),+])
    );
}

/// A macro for constructing buffers akin to `vec![..]`.
#[macro_export]
macro_rules! buffer_mut {
    () => (
        $crate::BufferMut::empty()
    );
    ($start:tt .. $end:tt) => (
        $crate::BufferMut::from_iter($start..$end)
    );
    ($start:tt ..= $end:tt) => (
        $crate::BufferMut::from_iter($start..=$end)
    );
    ($elem:expr; $n:expr) => (
        $crate::BufferMut::full($elem, $n)
    );
    ($($x:expr),+ $(,)?) => (
        $crate::BufferMut::from_iter([$($x),+])
    );
}
