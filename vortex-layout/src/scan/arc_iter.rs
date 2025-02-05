use std::sync::Arc;

/// There is no `IntoIterator` for `Arc<[T]>` so to avoid copying into a Vec<T>, we define our own.
/// See <https://users.rust-lang.org/t/arc-to-owning-iterator/115190/11>.
pub(crate) struct ArcIter<T> {
    inner: Arc<[T]>,
    pos: usize,
}

impl<T> ArcIter<T> {
    pub(crate) fn new(inner: Arc<[T]>) -> Self {
        Self { inner, pos: 0 }
    }
}

impl<T: Clone> Iterator for ArcIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        (self.pos < self.inner.len()).then(|| {
            let item = self.inner[self.pos].clone();
            self.pos += 1;
            item
        })
    }
}
