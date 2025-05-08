use crate::vtable::VTable;

/// Collection of functions required for arrays that can hold [`DType::Bool`] values.
pub trait BoolVTable<V: VTable> {}
