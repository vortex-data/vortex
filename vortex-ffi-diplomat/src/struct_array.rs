// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Diplomat bridge for the column-wise struct array builder.
//!
//! In the hand-written C ABI this was a `box_wrapper!(StructBuilder, vx_struct_column_builder)`
//! with `vx_struct_column_builder_new`, `vx_struct_column_builder_add_field`,
//! `vx_struct_column_builder_finalize`, and the macro-generated `_free`.
//!
//! Under Diplomat the builder is an opaque type with `&mut self` mutating methods. Diplomat
//! auto-generates the destructor (replacing `_free`). `add_field` returns `Result` instead of
//! using an `error_out` out-parameter, and `finalize` borrows `&self` (building from the current
//! state) rather than consuming a raw pointer — the handle is still dropped by the caller through
//! Diplomat's generated destructor, preserving the observable behaviour.

pub use ffi::VxStructColumnBuilder;

#[diplomat::bridge]
pub mod ffi {
    use vortex::array::arrays::StructArray;
    use vortex::array::validity::Validity;
    use vortex::array::{ArrayRef, IntoArray};
    use vortex::dtype::FieldName;
    use vortex::error::vortex_bail;

    use crate::array::ffi::{VxArray, VxValidityKind};
    use crate::error::ffi::VortexFfiError;

    /// A column-wise builder for struct arrays.
    ///
    /// Fields are added one at a time with [`Self::add_field`] and the result is produced with
    /// [`Self::finalize`]. Replaces the C `vx_struct_column_builder`.
    #[diplomat::opaque]
    pub struct VxStructColumnBuilder {
        names: Vec<FieldName>,
        fields: Vec<ArrayRef>,
        validity: Validity,
    }

    impl VxStructColumnBuilder {
        /// Create a new column-wise struct array builder.
        ///
        /// `capacity` is a hint for the number of columns (pass 0 if unknown). Replaces
        /// `vx_struct_column_builder_new`; the `Array` validity case is not supported here, in line
        /// with the C constructor which only consumed the validity descriptor's non-array kinds for
        /// the struct's own validity.
        #[diplomat::attr(auto, constructor)]
        pub fn new(validity: VxValidityKind, capacity: usize) -> Box<VxStructColumnBuilder> {
            Box::new(VxStructColumnBuilder {
                names: Vec::with_capacity(capacity),
                fields: Vec::with_capacity(capacity),
                validity: match validity {
                    VxValidityKind::NonNullable => Validity::NonNullable,
                    VxValidityKind::AllValid => Validity::AllValid,
                    VxValidityKind::AllInvalid => Validity::AllInvalid,
                },
            })
        }

        /// Add a named field to the builder.
        ///
        /// Errors if the field length does not match previously-added fields. On error the builder
        /// remains valid. Replaces `vx_struct_column_builder_add_field`.
        pub fn add_field(
            &mut self,
            name: &str,
            field: &VxArray,
        ) -> Result<(), Box<VortexFfiError>> {
            let field = field.0.clone();
            if !self.fields.is_empty() && field.len() != self.fields[0].len() {
                vortex_bail!(
                    "Field length mismatch: expected {}, got {}",
                    self.fields[0].len(),
                    field.len()
                );
            }
            self.names.push(FieldName::from(name));
            self.fields.push(field);
            Ok(())
        }

        /// Finalize the builder into a struct array.
        ///
        /// Replaces `vx_struct_column_builder_finalize`. Builds from the current set of fields;
        /// the builder handle is released by the caller through Diplomat's generated destructor.
        pub fn finalize(&self) -> Result<Box<VxArray>, Box<VortexFfiError>> {
            let rows = self.fields.first().map(|f| f.len()).unwrap_or(0);
            let array = StructArray::try_new(
                self.names.clone().into(),
                self.fields.clone(),
                rows,
                self.validity.clone(),
            )?;
            Ok(Box::new(VxArray(array.into_array())))
        }
    }
}
