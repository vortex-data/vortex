// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::cpp;
use crate::duckdb::LogicalType;
use crate::duckdb::SelectionVector;
use crate::duckdb::Vector;
use crate::duckdb::VectorRef;
use crate::lifetime_wrapper;

lifetime_wrapper!(
    /// A reusable dictionary buffer that can be used to efficiently create dictionary vectors.
    ///
    /// This wraps DuckDB's `buffer_ptr<VectorChildBuffer>` which is created by
    /// `DictionaryVector::CreateReusableDictionary` and can be reused with
    /// `Vector::Dictionary` to create dictionary vectors without copying the values.
    ReusableDict,
    cpp::duckdb_vx_reusable_dict,
    cpp::duckdb_vx_reusable_dict_destroy
);

impl ReusableDict {
    /// Create a new reusable dictionary from a logical type and size.
    ///
    /// The dictionary buffer can be reused with multiple selection vectors
    /// using `Vector::dictionary_reusable`.
    pub fn new(logical_type: LogicalType, size: usize) -> Self {
        unsafe {
            Self::own(cpp::duckdb_vx_reusable_dict_create(
                logical_type.as_ptr(),
                size as _,
            ))
        }
    }

    /// Get the internal vector of the reusable dictionary.
    ///
    /// The returned vector is borrowed from this reusable dictionary and can be used
    /// to populate the dictionary values.
    pub fn vector(&mut self) -> &mut VectorRef {
        let mut out_vector = std::ptr::null_mut();
        unsafe {
            cpp::duckdb_vx_reusable_dict_set_vector(self.as_ptr(), &raw mut out_vector);
            Vector::borrow_mut(out_vector)
        }
    }
}

impl Clone for ReusableDict {
    fn clone(&self) -> Self {
        unsafe { Self::own(cpp::duckdb_vx_reusable_dict_clone(self.as_ptr())) }
    }
}

/// Safety: ReusableDict wraps a DuckDB buffer_ptr which is reference-counted
/// and thread-safe.
unsafe impl Send for ReusableDict {}
unsafe impl Sync for ReusableDict {}

impl VectorRef {
    /// Creates a dictionary vector using a reusable dictionary and a selection vector.
    ///
    /// This is more efficient than `dictionary` when the same dictionary values are
    /// used multiple times with different selection vectors.
    pub fn reuse_dictionary(&mut self, reusable: &ReusableDict, sel_vec: &SelectionVector) {
        unsafe {
            cpp::duckdb_vx_vector_dictionary_reusable(
                self.as_ptr(),
                reusable.as_ptr(),
                sel_vec.as_ptr(),
            )
        }
    }
}
