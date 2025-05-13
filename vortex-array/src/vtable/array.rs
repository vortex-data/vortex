use vortex_dtype::DType;

use crate::stats::StatsSetRef;
use crate::vtable::VTable;

pub trait ArrayVTable<V: VTable> {
    fn len(array: &V::Array) -> usize;

    fn dtype(array: &V::Array) -> &DType;

    fn stats(array: &V::Array) -> StatsSetRef<'_>;
}
