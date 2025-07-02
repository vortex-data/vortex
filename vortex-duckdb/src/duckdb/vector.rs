use std::ffi::CStr;

use bitvec::slice::BitSlice;

use crate::duckdb::data::Data;
use crate::duckdb::{LogicalType, SelectionVector, Value};
use crate::{cpp, wrapper};

pub const DUCKDB_STANDARD_VECTOR_SIZE: usize = 2048;

wrapper!(Vector, cpp::duckdb_vector, cpp::duckdb_destroy_vector);

impl Vector {
    /// Create a new vector with the given type and capacity.
    pub fn with_capacity(logical_type: LogicalType, len: usize) -> Self {
        unsafe { Self::own(cpp::duckdb_create_vector(logical_type.as_ptr(), len as _)) }
    }

    /// Converts the vector to a constant value.
    pub fn reference_value(&mut self, value: &Value) {
        unsafe {
            cpp::duckdb_vector_reference_value(self.as_ptr(), value.as_ptr());
        }
    }

    /// Populates this vector by reference to another.
    pub fn reference(&mut self, other: &Vector) {
        unsafe { cpp::duckdb_vector_reference_vector(self.as_ptr(), other.as_ptr()) }
    }

    /// Slice the vector to a new dictionary vector, using the current vector's values and
    /// the provided selection vector.
    pub fn slice_to_dictionary(&mut self, sel_vec: SelectionVector, sel_vec_length: usize) {
        unsafe {
            cpp::duckdb_vx_vector_slice_to_dictionary(
                self.as_ptr(),
                sel_vec.as_ptr(),
                sel_vec_length as _,
            )
        }
    }

    pub fn to_sequence(&mut self, start: i64, stop: i64, capacity: u64) {
        unsafe { cpp::duckdb_vx_sequence_vector(self.ptr, start, stop, capacity) }
    }

    // NOTE(ngates): vector doesn't hold its own length. Which makes writing a safe
    //  Rust API annoying...
    pub unsafe fn as_slice_mut<T>(&mut self, length: usize) -> &mut [T] {
        let ptr = unsafe { cpp::duckdb_vector_get_data(self.as_ptr()) };
        unsafe { std::slice::from_raw_parts_mut(ptr.cast::<T>(), length) }
    }

    pub fn add_string_buffer<T>(&self, buffer: T) {
        let data = Data::from(Box::new(buffer));
        unsafe { cpp::duckdb_vx_string_vector_add_buffer(self.as_ptr(), data.into_ptr()) }
    }

    /// Assigns the element at the specified index with a string value.
    /// FIXME(ngates): remove this.
    pub fn assign_string_element(&self, idx: usize, value: &CStr) {
        unsafe { cpp::duckdb_vector_assign_string_element(self.as_ptr(), idx as _, value.as_ptr()) }
    }

    pub fn logical_type(&self) -> LogicalType {
        unsafe { LogicalType::own(cpp::duckdb_vector_get_column_type(self.as_ptr())) }
    }

    #[allow(clippy::expect_used)]
    pub fn ensure_validity_slice(&mut self) -> &mut BitSlice<u64> {
        unsafe { cpp::duckdb_vector_ensure_validity_writable(self.as_ptr()) };
        self.validity_slice_mut()
            .expect("we just ensured the validity slice is allocated")
    }

    /// Returns the validity slice of the vector, if it exists.
    pub fn validity_slice_mut(&mut self) -> Option<&mut BitSlice<u64>> {
        let ptr = unsafe { cpp::duckdb_vector_get_validity(self.as_ptr()) };
        unsafe { ptr.as_mut() }.map(|ptr| {
            let len = DUCKDB_STANDARD_VECTOR_SIZE / 64;
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
            BitSlice::from_slice_mut(slice)
        })
    }
}

impl Clone for Vector {
    fn clone(&self) -> Self {
        // Return an unowned copy of the vector
        unsafe { Vector::borrow(self.as_ptr()) }
    }
}
