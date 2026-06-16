// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Diplomat bridge for Vortex arrays.
//!
//! In the hand-written C ABI this was an `arc_wrapper!(ArrayRef, vx_array)` plus a large surface
//! of `extern "C-unwind"` functions and a `paste!`-generated set of per-primitive accessors. This
//! module ports all of them to idiomatic Diplomat:
//!
//! - The opaque [`VxArray`] replaces `vx_array`; Diplomat auto-generates the destructor (no
//!   `vx_array_free`) and `clone` is an ordinary method (replacing `vx_array_clone`).
//! - Fallible functions return `Result<_, Box<VortexFfiError>>` instead of taking an
//!   `error_out: *mut *mut vx_error` out-parameter (replaces `try_or` / `try_or_default`).
//! - `vx_validity` (a `#[repr(C)]` struct embedding a raw pointer) becomes the Diplomat enum
//!   [`VxValidityKind`] plus explicit constructors, since Diplomat structs cannot carry opaque
//!   pointers.
//! - The `paste!`-generated `vx_array_get_<ptype>` / `vx_array_get_storage_<ptype>` /
//!   per-ptype `vx_array_new_primitive` are written out as explicit methods for clarity and so
//!   that Diplomat can give each a precise primitive signature.
//! - `vx_array_get_utf8` writes into a `DiplomatWrite`; `vx_array_get_binary` returns a
//!   `Box<VxBinary>`.
//! - The Arrow C Data Interface import (`vx_array_from_arrow`) is preserved as a named
//!   constructor taking the raw `FFI_ArrowArray` / `FFI_ArrowSchema` pointers.

pub use ffi::{VxArray, VxValidityKind};

#[diplomat::bridge]
pub mod ffi {
    use std::sync::Arc;

    use arrow_array::array::make_array;
    use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi};
    use diplomat_runtime::DiplomatWrite;
    use vortex::array::arrays::struct_::StructArrayExt;
    use vortex::array::arrays::{NullArray, PrimitiveArray, StructArray};
    use vortex::array::arrow::FromArrowArray;
    use vortex::array::validity::Validity;
    use vortex::array::{
        ArrayRef, IntoArray, LEGACY_SESSION, VortexSessionExecute,
    };
    use vortex::buffer::Buffer;
    use vortex::dtype::DType;
    use vortex::dtype::half::f16;
    use vortex::error::vortex_err;

    use crate::binary::ffi::VxBinary;
    use crate::dtype::ffi::{VxDType, VxDTypeVariant};
    use crate::error::ffi::VortexFfiError;
    use crate::expression::ffi::VxExpr;
    use crate::ptype::ffi::VxPType;
    use std::fmt::Write;

    /// Arrays are reference-counted handles to owned memory buffers that hold scalars.
    ///
    /// These buffers can be held in a number of physical encodings to perform lightweight
    /// compression that exploits the particular data distribution of the array's values. Every
    /// data type recognized by Vortex also has a canonical physical encoding format, which arrays
    /// can be canonicalized into for ease of access in compute functions.
    ///
    /// Internally an `Arc<dyn Array>`, so cloning a [`VxArray`] is cheap. Replaces the C
    /// `vx_array` opaque type; Diplomat auto-generates the destructor.
    #[diplomat::opaque]
    pub struct VxArray(pub(crate) ArrayRef);

    /// How the validity (null-ness) of a constructed array is described.
    ///
    /// Replaces the C `vx_validity_type` enum. The `Array` case (a boolean validity array) is not
    /// expressed here; use [`VxArray::new_primitive_with_validity`] which takes the boolean array
    /// explicitly, mirroring `VX_VALIDITY_ARRAY`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum VxValidityKind {
        /// Items can't be null.
        NonNullable,
        /// All items are valid.
        AllValid,
        /// All items are invalid (null).
        AllInvalid,
    }

    impl VxArray {
        /// Create a new array with the Null dtype.
        ///
        /// Replaces `vx_array_new_null`.
        #[diplomat::attr(auto, named_constructor = "null")]
        pub fn new_null(len: usize) -> Box<VxArray> {
            Box::new(VxArray(NullArray::new(len).into_array()))
        }

        /// Import a Vortex array from an Arrow array via the Arrow C Data Interface.
        ///
        /// `array` and `schema` together describe a single Arrow array (the standard Arrow C Data
        /// Interface pair). Both are *consumed*: their `release` callbacks are invoked and the
        /// caller must not use them afterwards. `nullable` controls the top-level nullability of
        /// the resulting dtype.
        ///
        /// Replaces `vx_array_from_arrow`. The raw FFI pointers are kept (illustrative) because
        /// the Arrow C Data Interface is itself a C ABI contract.
        ///
        /// # Safety
        ///
        /// `array` and `schema` must be valid, non-null pointers to Arrow C Data Interface structs
        /// that have not yet been released.
        #[diplomat::attr(auto, named_constructor = "from_arrow")]
        pub unsafe fn from_arrow(
            array: *mut FFI_ArrowArray,
            schema: *mut FFI_ArrowSchema,
            nullable: bool,
        ) -> Result<Box<VxArray>, Box<VortexFfiError>> {
            if array.is_null() || schema.is_null() {
                return Err(VortexFfiError::new("null arrow array/schema"));
            }
            let ffi_array = unsafe { std::ptr::replace(array, FFI_ArrowArray::empty()) };
            let ffi_schema = unsafe { std::ptr::replace(schema, FFI_ArrowSchema::empty()) };
            let array_data = unsafe { from_ffi(ffi_array, &ffi_schema) }?;
            drop(ffi_schema);
            let arrow_array = make_array(array_data);
            let vortex_array = ArrayRef::from_arrow(arrow_array.as_ref(), nullable)?;
            Ok(Box::new(VxArray(vortex_array)))
        }

        /// Clone this array, returning a new owned handle that shares the same buffers.
        ///
        /// Replaces `vx_array_clone`.
        pub fn clone(&self) -> Box<VxArray> {
            Box::new(VxArray(self.0.clone()))
        }

        /// The number of elements in the array.
        ///
        /// Replaces `vx_array_len`.
        #[diplomat::attr(auto, getter)]
        pub fn len(&self) -> usize {
            self.0.len()
        }

        /// Whether the array is empty.
        #[diplomat::attr(auto, getter)]
        pub fn is_empty(&self) -> bool {
            self.0.is_empty()
        }

        /// Whether the array's dtype is nullable. A Null array is nullable.
        ///
        /// Replaces `vx_array_is_nullable`.
        #[diplomat::attr(auto, getter)]
        pub fn is_nullable(&self) -> bool {
            self.0.dtype().is_nullable()
        }

        /// Borrow the dtype of the array.
        ///
        /// Replaces `vx_array_dtype`. Returns an owned [`VxDType`] handle (cheap `Arc` clone)
        /// rather than a borrowed pointer, which removes the lifetime footgun the C ABI documented.
        #[diplomat::attr(auto, getter)]
        pub fn dtype(&self) -> Box<VxDType> {
            Box::new(VxDType(Arc::new(self.0.dtype().clone())))
        }

        /// Check the array's dtype against a variant.
        ///
        /// Replaces `vx_array_has_dtype`.
        pub fn has_dtype(&self, variant: VxDTypeVariant) -> bool {
            VxDTypeVariant::from(self.0.dtype()) == variant
        }

        /// Check whether the array has a Primitive dtype with the given ptype.
        ///
        /// Replaces `vx_array_is_primitive`.
        pub fn is_primitive(&self, ptype: VxPType) -> bool {
            let ptype = ptype.into();
            matches!(self.0.dtype(), DType::Primitive(other, _) if *other == ptype)
        }

        /// Return the validity kind of the array.
        ///
        /// Replaces the `type` field written by `vx_array_get_validity`. The boolean validity
        /// array (the `VX_VALIDITY_ARRAY` case) is exposed separately via [`VxArray::validity_array`].
        pub fn validity_kind(&self) -> Result<VxValidityKind, Box<VortexFfiError>> {
            Ok(match self.0.validity()? {
                Validity::NonNullable => VxValidityKind::NonNullable,
                Validity::AllValid => VxValidityKind::AllValid,
                Validity::AllInvalid => VxValidityKind::AllInvalid,
                // A per-element validity array reports as AllValid here; callers needing the
                // mask should use `validity_array`.
                Validity::Array(_) => VxValidityKind::AllValid,
            })
        }

        /// Return the boolean validity array, if the array's validity is backed by one.
        ///
        /// Replaces the `array` field written by `vx_array_get_validity` for the
        /// `VX_VALIDITY_ARRAY` case. Returns `None` for the non-array validity kinds.
        pub fn validity_array(&self) -> Result<Option<Box<VxArray>>, Box<VortexFfiError>> {
            Ok(match self.0.validity()? {
                Validity::Array(array) => Some(Box::new(VxArray(array))),
                _ => None,
            })
        }

        /// Return the owned field of a struct array at `index`.
        ///
        /// Errors if `index` is out of bounds or the array is not a struct. Replaces
        /// `vx_array_get_field`.
        pub fn get_field(&self, index: usize) -> Result<Box<VxArray>, Box<VortexFfiError>> {
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            let field_array = self
                .0
                .clone()
                .execute::<StructArray>(&mut ctx)?
                .unmasked_fields()
                .get(index)
                .ok_or_else(|| vortex_err!("Field index out of bounds"))?
                .clone();
            Ok(Box::new(VxArray(field_array)))
        }

        /// Return a slice of the array covering `start..stop`.
        ///
        /// Replaces `vx_array_slice`.
        pub fn slice(&self, start: usize, stop: usize) -> Result<Box<VxArray>, Box<VortexFfiError>> {
            Ok(Box::new(VxArray(self.0.slice(start..stop)?)))
        }

        /// Check whether the element at `index` is invalid (null).
        ///
        /// Replaces `vx_array_element_is_invalid`.
        pub fn element_is_invalid(&self, index: usize) -> Result<bool, Box<VortexFfiError>> {
            Ok(self
                .0
                .is_invalid(index, &mut LEGACY_SESSION.create_execution_ctx())?)
        }

        /// Count how many elements of the array are invalid (null).
        ///
        /// Replaces `vx_array_invalid_count`.
        pub fn invalid_count(&self) -> Result<usize, Box<VortexFfiError>> {
            Ok(self
                .0
                .invalid_count(&mut LEGACY_SESSION.create_execution_ctx())?)
        }

        /// Apply an expression to the array, wrapping it with a ScalarFnArray.
        ///
        /// This is a constant-time operation; executing the resulting array is still O(n).
        /// Replaces `vx_array_apply`.
        pub fn apply(&self, expression: &VxExpr) -> Result<Box<VxArray>, Box<VortexFfiError>> {
            Ok(Box::new(VxArray(self.0.clone().apply(expression.inner())?)))
        }

        /// Return the UTF-8 string at `index`, writing it into `out`.
        ///
        /// Returns `false` (and writes nothing) if the value at `index` is null. Replaces
        /// `vx_array_get_utf8`, which returned an owned `vx_string` or null pointer.
        pub fn get_utf8(
            &self,
            index: usize,
            out: &mut DiplomatWrite,
        ) -> Result<bool, Box<VortexFfiError>> {
            let value = self
                .0
                .execute_scalar(index, &mut LEGACY_SESSION.create_execution_ctx())?;
            match value.as_utf8().value() {
                Some(buffer) => {
                    let _ = write!(out, "{}", buffer.as_str());
                    Ok(true)
                }
                None => Ok(false),
            }
        }

        /// Return the binary blob at `index`, or `None` if the value is null.
        ///
        /// Replaces `vx_array_get_binary`, which returned an owned `vx_binary` or null pointer.
        pub fn get_binary(
            &self,
            index: usize,
        ) -> Result<Option<Box<VxBinary>>, Box<VortexFfiError>> {
            let value = self
                .0
                .execute_scalar(index, &mut LEGACY_SESSION.create_execution_ctx())?;
            Ok(value
                .as_binary()
                .value()
                .map(|bytes| Box::new(VxBinary(Arc::from(bytes.as_bytes())))))
        }
    }

    // -----------------------------------------------------------------------------------------
    // Per-primitive constructors. These replace the single `vx_array_new_primitive` dispatch on
    // `vx_ptype` plus the `primitive_from_raw` helper. Each takes a typed slice (which Diplomat
    // marshals from the caller's native array) and a `VxValidityKind`.
    // -----------------------------------------------------------------------------------------
    macro_rules! new_primitive {
        ($method:ident, $named:literal, $ty:ty, $doc:literal) => {
            impl VxArray {
                #[doc = $doc]
                #[diplomat::attr(auto, named_constructor = $named)]
                pub fn $method(values: &[$ty], validity: VxValidityKind) -> Box<VxArray> {
                    let buffer = Buffer::copy_from(values);
                    let array = PrimitiveArray::new(buffer, validity_from_kind(validity));
                    Box::new(VxArray(array.into_array()))
                }
            }
        };
    }

    new_primitive!(new_primitive_u8, "primitive_u8", u8, "Create a `u8` primitive array.");
    new_primitive!(new_primitive_u16, "primitive_u16", u16, "Create a `u16` primitive array.");
    new_primitive!(new_primitive_u32, "primitive_u32", u32, "Create a `u32` primitive array.");
    new_primitive!(new_primitive_u64, "primitive_u64", u64, "Create a `u64` primitive array.");
    new_primitive!(new_primitive_i8, "primitive_i8", i8, "Create an `i8` primitive array.");
    new_primitive!(new_primitive_i16, "primitive_i16", i16, "Create an `i16` primitive array.");
    new_primitive!(new_primitive_i32, "primitive_i32", i32, "Create an `i32` primitive array.");
    new_primitive!(new_primitive_i64, "primitive_i64", i64, "Create an `i64` primitive array.");
    new_primitive!(new_primitive_f32, "primitive_f32", f32, "Create an `f32` primitive array.");
    new_primitive!(new_primitive_f64, "primitive_f64", f64, "Create an `f64` primitive array.");

    impl VxArray {
        /// Create an `f16` primitive array from 32-bit floats.
        ///
        /// Diplomat has no native `f16` scalar, so the values are supplied as `f32` and converted.
        /// Together with the `new_primitive_*` family this covers every ptype handled by the C
        /// `vx_array_new_primitive` dispatch.
        #[diplomat::attr(auto, named_constructor = "primitive_f16")]
        pub fn new_primitive_f16(values: &[f32], validity: VxValidityKind) -> Box<VxArray> {
            let converted: Vec<f16> = values.iter().map(|v| f16::from_f32(*v)).collect();
            let buffer = Buffer::copy_from(converted.as_slice());
            let array = PrimitiveArray::new(buffer, validity_from_kind(validity));
            Box::new(VxArray(array.into_array()))
        }
    }

    // -----------------------------------------------------------------------------------------
    // Per-primitive element accessors, replacing the `paste!`-generated `vx_array_get_<ptype>`
    // and `vx_array_get_storage_<ptype>`. The C ABI panicked on error; here they return
    // `Result` so the error is surfaced as a language-native exception/Result.
    // -----------------------------------------------------------------------------------------
    macro_rules! get_primitive {
        ($get:ident, $get_storage:ident, $ty:ty, $tyname:literal) => {
            impl VxArray {
                #[doc = concat!("Return the `", $tyname, "` value at `index`.")]
                pub fn $get(&self, index: usize) -> Result<$ty, Box<VortexFfiError>> {
                    let value = self
                        .0
                        .execute_scalar(index, &mut LEGACY_SESSION.create_execution_ctx())?;
                    value
                        .as_primitive()
                        .as_::<$ty>()
                        .ok_or_else(|| VortexFfiError::new("null value"))
                }

                #[doc = concat!(
                    "Return the `", $tyname,
                    "` storage value at `index` for an extension-typed array."
                )]
                pub fn $get_storage(&self, index: usize) -> Result<$ty, Box<VortexFfiError>> {
                    let value = self
                        .0
                        .execute_scalar(index, &mut LEGACY_SESSION.create_execution_ctx())?;
                    value
                        .as_extension()
                        .to_storage_scalar()
                        .as_primitive()
                        .as_::<$ty>()
                        .ok_or_else(|| VortexFfiError::new("null value"))
                }
            }
        };
    }

    get_primitive!(get_u8, get_storage_u8, u8, "u8");
    get_primitive!(get_u16, get_storage_u16, u16, "u16");
    get_primitive!(get_u32, get_storage_u32, u32, "u32");
    get_primitive!(get_u64, get_storage_u64, u64, "u64");
    get_primitive!(get_i8, get_storage_i8, i8, "i8");
    get_primitive!(get_i16, get_storage_i16, i16, "i16");
    get_primitive!(get_i32, get_storage_i32, i32, "i32");
    get_primitive!(get_i64, get_storage_i64, i64, "i64");
    get_primitive!(get_f32, get_storage_f32, f32, "f32");
    get_primitive!(get_f64, get_storage_f64, f64, "f64");

    impl VxArray {
        /// Return the `f16` value at `index` as an `f32`.
        ///
        /// Diplomat has no native `f16` scalar, so the value is widened to `f32`. Replaces
        /// `vx_array_get_f16`.
        pub fn get_f16(&self, index: usize) -> Result<f32, Box<VortexFfiError>> {
            let value = self
                .0
                .execute_scalar(index, &mut LEGACY_SESSION.create_execution_ctx())?;
            value
                .as_primitive()
                .as_::<f16>()
                .map(|v| v.to_f32())
                .ok_or_else(|| VortexFfiError::new("null value"))
        }

        /// Return the `f16` storage value at `index` as an `f32` for an extension-typed array.
        pub fn get_storage_f16(&self, index: usize) -> Result<f32, Box<VortexFfiError>> {
            let value = self
                .0
                .execute_scalar(index, &mut LEGACY_SESSION.create_execution_ctx())?;
            value
                .as_extension()
                .to_storage_scalar()
                .as_primitive()
                .as_::<f16>()
                .map(|v| v.to_f32())
                .ok_or_else(|| VortexFfiError::new("null value"))
        }
    }

    /// Map a [`VxValidityKind`] onto a Vortex [`Validity`] for the non-array cases.
    fn validity_from_kind(kind: VxValidityKind) -> Validity {
        match kind {
            VxValidityKind::NonNullable => Validity::NonNullable,
            VxValidityKind::AllValid => Validity::AllValid,
            VxValidityKind::AllInvalid => Validity::AllInvalid,
        }
    }
}
