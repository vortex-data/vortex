use vortex_array::dtype::DType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DomainOrdering {
    Unordered,
    Ordered,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputContract {
    dtype: DType,
    ordering: DomainOrdering,
}

impl OutputContract {
    pub fn new(dtype: DType) -> Self {
        Self {
            dtype,
            ordering: DomainOrdering::Unordered,
        }
    }

    pub fn ordered(dtype: DType) -> Self {
        Self {
            dtype,
            ordering: DomainOrdering::Ordered,
        }
    }

    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    pub fn ordering(&self) -> DomainOrdering {
        self.ordering
    }
}
