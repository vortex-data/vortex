// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// A macro for constructing bit-buffers akin to `vec![..]`.
#[macro_export]
macro_rules! bitbuffer {
    () => (
        $crate::BitBuffer::empty()
    );
    ($elem:expr; $n:expr) => (
        $crate::BitBuffer::full($elem, $n)
    );
    ($($x:expr),+ $(,)?) => (
        $crate::BitBuffer::from_iter([$($x),+])
    );
}

/// A macro for constructing bit-buffers akin to `vec![..]`.
#[macro_export]
macro_rules! bitbuffer_mut {
    () => (
        $crate::BitBufferMut::empty()
    );
    ($elem:expr; $n:expr) => (
        $crate::BitBufferMut::full($elem, $n)
    );
    ($($x:expr),+ $(,)?) => (
        $crate::BitBufferMut::from_iter([$($x),+])
    );
}
