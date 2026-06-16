// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Diplomat bridge for Vortex logical data types (`DType`).
//!
//! This replaces the hand-written `vx_dtype` arc-wrapper and its free-standing
//! `extern "C-unwind"` constructors/accessors. The opaque `VxDType` wraps an
//! `Arc<DType>`; Diplomat auto-generates the destructor, so there is no manual
//! `vx_dtype_free` / `vx_dtype_clone` here (the C ABI provided both via `arc_wrapper!`).
//!
//! Notable ABI mapping differences:
//! - C constructors returned `*const vx_dtype` and "took ownership" of child dtype pointers.
//!   In Diplomat, owned returns are `Box<VxDType>` and child dtypes are passed as
//!   `Box<VxDType>` (Diplomat consumes boxed-opaque arguments), so ownership transfer is
//!   expressed in the type system rather than documented.
//! - The C "borrowed reference" accessors (`vx_dtype_list_element`, `vx_dtype_struct_dtype`,
//!   ...) returned raw pointers tied to the parent's lifetime. Diplomat models these as
//!   `&'a self -> Box<...>` owned clones, since Diplomat cannot express a borrow whose
//!   lifetime is tied to a returned opaque. Cloning an `Arc<DType>` is cheap.
//! - Functions that the C ABI made fallible via an `error_out` out-parameter
//!   (`vx_dtype_to_arrow_schema`, `vx_dtype_from_arrow_schema`) now return
//!   `Result<_, Box<VortexFfiError>>`.
//! - C accessors that `vortex_panic!`/`vortex_expect`-ed on the wrong variant
//!   (decimal precision/scale, fsl element/size, time unit/zone) are made fallible
//!   `Result<_, Box<VortexFfiError>>` here, which is the idiomatic Diplomat equivalent.

pub use ffi::{VxDType, VxDTypeVariant};

#[diplomat::bridge]
pub mod ffi {
    use std::fmt::Write;
    use std::sync::Arc;

    use diplomat_runtime::DiplomatWrite;
    use vortex::dtype::DType;
    use vortex::dtype::DecimalDType;
    use vortex::extension::datetime::AnyTemporal;
    use vortex::extension::datetime::Date;
    use vortex::extension::datetime::Time;
    use vortex::extension::datetime::Timestamp;

    use crate::error::ffi::VortexFfiError;
    use crate::ptype::VxPType;
    use crate::struct_fields::ffi::VxStructFields;

    /// A Vortex data type.
    ///
    /// Data types in Vortex are purely logical, meaning they confer no information about how
    /// the data is physically stored. Replaces the C `vx_dtype` arc-wrapper.
    #[diplomat::opaque]
    pub struct VxDType(pub(crate) Arc<DType>);

    /// The variant tag of a [`VxDType`].
    ///
    /// Replaces the C `vx_dtype_variant` `#[repr(C)]` enum.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum VxDTypeVariant {
        /// Null type.
        Null,
        /// Boolean type.
        Bool,
        /// Primitive types (e.g. u8, i16, f32, ...).
        Primitive,
        /// Variable-length UTF-8 string type.
        Utf8,
        /// Variable-length binary data type.
        Binary,
        /// Nested struct type.
        Struct,
        /// Nested list type.
        List,
        /// User-defined extension type.
        Extension,
        /// Decimal type with fixed precision and scale.
        Decimal,
        /// Nested fixed-size list type.
        FixedSizeList,
    }

    impl VxDType {
        /// Create a new null data type.
        #[diplomat::attr(auto, named_constructor = "null")]
        pub fn new_null() -> Box<VxDType> {
            Box::new(VxDType(Arc::new(DType::Null)))
        }

        /// Create a new boolean data type.
        #[diplomat::attr(auto, named_constructor = "bool")]
        pub fn new_bool(is_nullable: bool) -> Box<VxDType> {
            Box::new(VxDType(Arc::new(DType::Bool(is_nullable.into()))))
        }

        /// Create a new primitive data type.
        #[diplomat::attr(auto, named_constructor = "primitive")]
        pub fn new_primitive(ptype: VxPType, is_nullable: bool) -> Box<VxDType> {
            Box::new(VxDType(Arc::new(DType::Primitive(
                ptype.into(),
                is_nullable.into(),
            ))))
        }

        /// Create a new variable-length UTF-8 data type.
        #[diplomat::attr(auto, named_constructor = "utf8")]
        pub fn new_utf8(is_nullable: bool) -> Box<VxDType> {
            Box::new(VxDType(Arc::new(DType::Utf8(is_nullable.into()))))
        }

        /// Create a new variable-length binary data type.
        #[diplomat::attr(auto, named_constructor = "binary")]
        pub fn new_binary(is_nullable: bool) -> Box<VxDType> {
            Box::new(VxDType(Arc::new(DType::Binary(is_nullable.into()))))
        }

        /// Create a new list data type with the given element type.
        ///
        /// Consumes `element`.
        #[diplomat::attr(auto, named_constructor = "list")]
        pub fn new_list(element: Box<VxDType>, is_nullable: bool) -> Box<VxDType> {
            Box::new(VxDType(Arc::new(DType::List(
                element.0,
                is_nullable.into(),
            ))))
        }

        /// Create a new fixed-size list data type.
        ///
        /// Consumes `element`.
        #[diplomat::attr(auto, named_constructor = "fixed_size_list")]
        pub fn new_fixed_size_list(
            element: Box<VxDType>,
            size: u32,
            is_nullable: bool,
        ) -> Box<VxDType> {
            Box::new(VxDType(Arc::new(DType::FixedSizeList(
                element.0,
                size,
                is_nullable.into(),
            ))))
        }

        /// Create a new struct data type from the given fields.
        ///
        /// Consumes `fields`.
        #[diplomat::attr(auto, named_constructor = "struct")]
        pub fn new_struct(fields: Box<VxStructFields>, is_nullable: bool) -> Box<VxDType> {
            Box::new(VxDType(Arc::new(DType::Struct(
                fields.0,
                is_nullable.into(),
            ))))
        }

        /// Create a new decimal data type.
        #[diplomat::attr(auto, named_constructor = "decimal")]
        pub fn new_decimal(precision: u8, scale: i8, is_nullable: bool) -> Box<VxDType> {
            Box::new(VxDType(Arc::new(DType::Decimal(
                DecimalDType::new(precision, scale),
                is_nullable.into(),
            ))))
        }

        /// The variant tag of this data type.
        #[diplomat::attr(auto, getter)]
        pub fn variant(&self) -> VxDTypeVariant {
            match self.0.as_ref() {
                DType::Null => VxDTypeVariant::Null,
                DType::Bool(_) => VxDTypeVariant::Bool,
                DType::Primitive(..) => VxDTypeVariant::Primitive,
                DType::Decimal(..) => VxDTypeVariant::Decimal,
                DType::Utf8(_) => VxDTypeVariant::Utf8,
                DType::Binary(_) => VxDTypeVariant::Binary,
                DType::List(..) => VxDTypeVariant::List,
                DType::FixedSizeList(..) => VxDTypeVariant::FixedSizeList,
                DType::Struct(..) => VxDTypeVariant::Struct,
                DType::Extension(_) => VxDTypeVariant::Extension,
                // Union / Variant are not yet exposed across the FFI boundary.
                _ => VxDTypeVariant::Extension,
            }
        }

        /// Whether this data type is nullable.
        #[diplomat::attr(auto, getter)]
        pub fn is_nullable(&self) -> bool {
            self.0.is_nullable()
        }

        /// The primitive type of a primitive data type.
        ///
        /// Errors if this is not a primitive data type.
        pub fn primitive_ptype(&self) -> Result<VxPType, Box<VortexFfiError>> {
            match self.0.as_ref() {
                DType::Primitive(ptype, _) => Ok((*ptype).into()),
                _ => Err(VortexFfiError::new("not a primitive dtype")),
            }
        }

        /// The precision of a decimal data type.
        ///
        /// Errors if this is not a decimal data type.
        pub fn decimal_precision(&self) -> Result<u8, Box<VortexFfiError>> {
            self.0
                .as_decimal_opt()
                .map(|d| d.precision())
                .ok_or_else(|| VortexFfiError::new("not a decimal dtype"))
        }

        /// The scale of a decimal data type.
        ///
        /// Errors if this is not a decimal data type.
        pub fn decimal_scale(&self) -> Result<i8, Box<VortexFfiError>> {
            self.0
                .as_decimal_opt()
                .map(|d| d.scale())
                .ok_or_else(|| VortexFfiError::new("not a decimal dtype"))
        }

        /// The fields of a struct data type.
        ///
        /// Returns an owned clone (the C ABI returned a borrowed pointer). Errors if this is
        /// not a struct data type.
        pub fn struct_fields(&self) -> Result<Box<VxStructFields>, Box<VortexFfiError>> {
            self.0
                .as_struct_fields_opt()
                .map(|fields| Box::new(VxStructFields(fields.clone())))
                .ok_or_else(|| VortexFfiError::new("not a struct dtype"))
        }

        /// The element type of a list data type.
        ///
        /// Returns an owned clone. Errors if this is not a list data type.
        pub fn list_element(&self) -> Result<Box<VxDType>, Box<VortexFfiError>> {
            match self.0.as_ref() {
                DType::List(element, _) => Ok(Box::new(VxDType(element.clone()))),
                _ => Err(VortexFfiError::new("not a list dtype")),
            }
        }

        /// The element type of a fixed-size list data type.
        ///
        /// Returns an owned clone. Errors if this is not a fixed-size list data type.
        pub fn fixed_size_list_element(&self) -> Result<Box<VxDType>, Box<VortexFfiError>> {
            match self.0.as_ref() {
                DType::FixedSizeList(element, _, _) => Ok(Box::new(VxDType(element.clone()))),
                _ => Err(VortexFfiError::new("not a fixed-size list dtype")),
            }
        }

        /// The fixed width of a fixed-size list data type.
        ///
        /// Errors if this is not a fixed-size list data type.
        pub fn fixed_size_list_size(&self) -> Result<u32, Box<VortexFfiError>> {
            match self.0.as_ref() {
                DType::FixedSizeList(_, size, _) => Ok(*size),
                _ => Err(VortexFfiError::new("not a fixed-size list dtype")),
            }
        }

        /// Whether this is a temporal `Time` extension type.
        #[diplomat::attr(auto, getter)]
        pub fn is_time(&self) -> bool {
            matches!(self.0.as_ref(), DType::Extension(ext) if ext.is::<Time>())
        }

        /// Whether this is a temporal `Date` extension type.
        #[diplomat::attr(auto, getter)]
        pub fn is_date(&self) -> bool {
            matches!(self.0.as_ref(), DType::Extension(ext) if ext.is::<Date>())
        }

        /// Whether this is a temporal `Timestamp` extension type.
        #[diplomat::attr(auto, getter)]
        pub fn is_timestamp(&self) -> bool {
            matches!(self.0.as_ref(), DType::Extension(ext) if ext.is::<Timestamp>())
        }

        /// The time unit of a temporal data type, encoded as a small integer.
        ///
        /// Errors if this is not a temporal data type.
        pub fn time_unit(&self) -> Result<u8, Box<VortexFfiError>> {
            let DType::Extension(ext) = self.0.as_ref() else {
                return Err(VortexFfiError::new("not a temporal dtype"));
            };
            ext.metadata_opt::<AnyTemporal>()
                .map(|opts| opts.time_unit().into())
                .ok_or_else(|| VortexFfiError::new("not temporal metadata"))
        }

        /// Write the timezone of a timestamp data type into `out`.
        ///
        /// In the C ABI this returned an owned `vx_string` (or NULL). Here the timezone string
        /// is written into `out`; an absent timezone writes nothing. Errors if this is not a
        /// timestamp data type.
        pub fn time_zone(&self, out: &mut DiplomatWrite) -> Result<(), Box<VortexFfiError>> {
            let DType::Extension(ext) = self.0.as_ref() else {
                return Err(VortexFfiError::new("not a timestamp dtype"));
            };
            let opts = ext
                .metadata_opt::<Timestamp>()
                .ok_or_else(|| VortexFfiError::new("not a timestamp dtype"))?;
            if let Some(zone) = opts.tz.as_ref() {
                let _ = write!(out, "{zone}");
            }
            Ok(())
        }

        /// Export this data type into an Arrow C Data Interface schema.
        ///
        /// The C ABI took a `*mut FFI_ArrowSchema` out-parameter and an `error_out` and
        /// returned an `int` status code. Diplomat has no first-class binding for the Arrow C
        /// Data Interface, so the destination `FFI_ArrowSchema` is passed as a raw integer
        /// address (`schema_ptr`) supplied by the caller's Arrow bindings, and failure is
        /// reported via the returned `Result`.
        pub fn to_arrow_schema(&self, schema_ptr: usize) -> Result<(), Box<VortexFfiError>> {
            use arrow_array::ffi::FFI_ArrowSchema;
            let arrow_schema = self.0.to_arrow_schema()?;
            let ffi_schema = FFI_ArrowSchema::try_from(&arrow_schema)?;
            // SAFETY: `schema_ptr` must be a valid, writable `*mut FFI_ArrowSchema` provided by
            // the caller's Arrow C Data Interface bindings.
            unsafe { std::ptr::write(schema_ptr as *mut FFI_ArrowSchema, ffi_schema) };
            Ok(())
        }

        /// Import a Vortex data type from an Arrow C Data Interface schema.
        ///
        /// `schema_ptr` is a raw integer address of an `FFI_ArrowSchema` describing a struct
        /// (record-batch) schema. The schema is consumed: its `release` callback is invoked
        /// and the caller must not use it afterwards. The result is a non-nullable struct
        /// data type, mirroring how Arrow record batches map to Vortex arrays.
        #[diplomat::attr(auto, named_constructor = "from_arrow_schema")]
        pub fn from_arrow_schema(schema_ptr: usize) -> Result<Box<VxDType>, Box<VortexFfiError>> {
            use arrow_array::ffi::FFI_ArrowSchema;
            use arrow_schema::Schema;
            use vortex::dtype::arrow::FromArrowType;

            if schema_ptr == 0 {
                return Err(VortexFfiError::new("null arrow schema"));
            }
            // SAFETY: `schema_ptr` must be a valid `*mut FFI_ArrowSchema`; we replace it with an
            // empty schema so the caller's pointer is left releasable/inert.
            let ffi_schema = unsafe {
                std::ptr::replace(schema_ptr as *mut FFI_ArrowSchema, FFI_ArrowSchema::empty())
            };
            let arrow_schema = Schema::try_from(&ffi_schema)?;
            drop(ffi_schema);
            Ok(Box::new(VxDType(Arc::new(DType::from_arrow(&arrow_schema)))))
        }
    }
}
