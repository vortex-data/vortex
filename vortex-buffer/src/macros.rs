/// A macro for constructing buffers akin to `vec![..]`.
#[macro_export]
macro_rules! buffer {
    () => (
        $crate::Buffer:empty()
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
        $crate::BufferMut:empty()
    );
    ($elem:expr; $n:expr) => (
        $crate::BufferMut::full($elem, $n)
    );
    ($($x:expr),+ $(,)?) => (
        $crate::BufferMut::from_iter([$($x),+])
    );
}
