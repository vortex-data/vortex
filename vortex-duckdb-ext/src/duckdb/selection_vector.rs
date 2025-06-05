use crate::{cpp, wrapper};

wrapper!(
    SelectionVector,
    cpp::duckdb_selection_vector,
    |ptr: &mut cpp::duckdb_selection_vector| {
        unsafe { cpp::duckdb_destroy_selection_vector(*ptr) }
    }
);

impl SelectionVector {
    pub fn with_capacity(len: usize) -> Self {
        unsafe { Self::from_owned(cpp::duckdb_create_selection_vector(len as _)) }
    }

    // NOTE(ngates): selection vector doesn't hold its own length. Which makes writing a safe
    //  Rust API annoying...
    pub unsafe fn as_slice_mut(&mut self, length: usize) -> &mut [u32] {
        let ptr = unsafe { cpp::duckdb_selection_vector_get_data_ptr(self.as_ptr()) };
        unsafe { std::slice::from_raw_parts_mut(ptr, length) }
    }
}
