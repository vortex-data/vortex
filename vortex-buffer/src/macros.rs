// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
