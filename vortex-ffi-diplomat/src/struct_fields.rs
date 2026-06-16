// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Diplomat bridge for Vortex struct field sets (`StructFields`).
//!
//! Replaces the hand-written `vx_struct_fields` box-wrapper, its accessors, and the
//! `vx_struct_fields_builder` box-wrapper. Diplomat auto-generates destructors, so the C
//! `*_free` functions are gone.
//!
//! Notable ABI mapping differences:
//! - `vx_struct_fields_nfields` returned `u64`; here `nfields` is a getter returning `u64`.
//! - `vx_struct_fields_field_name` returned a borrowed `vx_string`; here `field_name` writes
//!   into a `DiplomatWrite` and returns a `Result` (out-of-bounds is an error rather than a
//!   NULL return).
//! - `vx_struct_fields_field_dtype` returned an owned `vx_dtype` (NULL on out-of-bounds /
//!   parse failure); here `field_dtype` returns `Result<Box<VxDType>, _>`.
//! - The builder's `add_field` "took ownership" of raw `vx_string` / `vx_dtype` pointers.
//!   In Diplomat, the name is passed as a `&DiplomatStr` (copied) and the dtype as a consumed
//!   `Box<VxDType>`. `finalize` consumes the builder (`self: Box<Self>`).

pub use ffi::{VxStructFields, VxStructFieldsBuilder};

#[diplomat::bridge]
pub mod ffi {
    use std::fmt::Write;
    use std::sync::Arc;

    use diplomat_runtime::{DiplomatStr, DiplomatWrite};
    use vortex::dtype::DType;
    use vortex::dtype::StructFields;

    use crate::dtype::ffi::VxDType;
    use crate::error::ffi::VortexFfiError;

    /// A Vortex struct data type's field set, without top-level nullability.
    ///
    /// Replaces the C `vx_struct_fields` box-wrapper.
    #[diplomat::opaque]
    pub struct VxStructFields(pub(crate) StructFields);

    impl VxStructFields {
        /// The number of fields.
        #[diplomat::attr(auto, getter)]
        pub fn nfields(&self) -> u64 {
            self.0.nfields() as u64
        }

        /// Write the name of the field at `idx` into `out`.
        ///
        /// Errors if `idx` is out of bounds.
        pub fn field_name(
            &self,
            idx: usize,
            out: &mut DiplomatWrite,
        ) -> Result<(), Box<VortexFfiError>> {
            if idx >= self.0.nfields() {
                return Err(VortexFfiError::new("field index out of bounds"));
            }
            let name = self.0.names()[idx].inner();
            let _ = write!(out, "{name}");
            Ok(())
        }

        /// The data type of the field at `idx`.
        ///
        /// Returns an owned dtype, since struct fields may be lazily parsed from a binary
        /// format. Errors if `idx` is out of bounds or the field dtype cannot be parsed.
        pub fn field_dtype(&self, idx: usize) -> Result<Box<VxDType>, Box<VortexFfiError>> {
            if idx >= self.0.nfields() {
                return Err(VortexFfiError::new("field index out of bounds"));
            }
            self.0
                .field_by_index(idx)
                .map(|field| Box::new(VxDType(Arc::new(field))))
                .ok_or_else(|| VortexFfiError::new("could not parse field dtype"))
        }
    }

    /// Builder for incrementally constructing a [`VxStructFields`].
    ///
    /// Replaces the C `vx_struct_fields_builder` box-wrapper.
    #[diplomat::opaque]
    pub struct VxStructFieldsBuilder {
        names: Vec<Arc<str>>,
        fields: Vec<DType>,
    }

    impl VxStructFieldsBuilder {
        /// Create a new, empty struct fields builder.
        #[diplomat::attr(auto, constructor)]
        pub fn new() -> Box<VxStructFieldsBuilder> {
            Box::new(VxStructFieldsBuilder {
                names: Vec::new(),
                fields: Vec::new(),
            })
        }

        /// Add a field with the given `name` and `dtype`.
        ///
        /// The name is copied; `dtype` is consumed.
        pub fn add_field(&mut self, name: &DiplomatStr, dtype: Box<VxDType>) {
            let name = String::from_utf8_lossy(name);
            self.names.push(Arc::from(name.as_ref()));
            self.fields.push(dtype.0.as_ref().clone());
        }

        /// Finalize the builder into a [`VxStructFields`], consuming it.
        pub fn finalize(self: Box<Self>) -> Box<VxStructFields> {
            let VxStructFieldsBuilder { names, fields } = *self;
            Box::new(VxStructFields(StructFields::new(names.into(), fields)))
        }
    }
}
