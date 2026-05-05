// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrowDTypeConverter`] and [`ArrowDTypeReader`] — pluggable DType ↔ Arrow conversion.

use std::fmt::Debug;
use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use vortex_error::VortexResult;

use crate::dtype::DType;
use crate::dtype::extension::ExtDTypeRef;

/// Reference-counted pointer to an [`ArrowDTypeConverter`].
pub type ArrowDTypeConverterRef = Arc<dyn ArrowDTypeConverter>;

/// Reference-counted pointer to an [`ArrowDTypeReader`].
pub type ArrowDTypeReaderRef = Arc<dyn ArrowDTypeReader>;

/// Plugin trait that converts a Vortex extension [`crate::dtype::DType`] into an Arrow
/// [`DataType`].
///
/// Converters are registered against an [`crate::dtype::extension::ExtId`] on the
/// [`crate::arrow::ArrowSession`]. Each extension type that wants Arrow representation must
/// register a converter; the dispatcher hard-fails on unknown extensions.
///
/// Implementations may also produce a full [`Field`] — useful when the Arrow representation
/// needs `ARROW:extension:name` metadata (for example `arrow.parquet.variant`).
pub trait ArrowDTypeConverter: 'static + Send + Sync + Debug {
    /// Convert the extension dtype to its Arrow [`DataType`].
    fn to_arrow_data_type(&self, ext: &ExtDTypeRef) -> VortexResult<DataType>;

    /// Convert the extension dtype to a fully-decorated Arrow [`Field`].
    ///
    /// The default implementation builds a `Field` from `name`, the result of
    /// [`ArrowDTypeConverter::to_arrow_data_type`], and `ext.is_nullable()`. Override when the
    /// extension type needs Arrow extension metadata on the field.
    fn to_arrow_field(&self, ext: &ExtDTypeRef, name: &str) -> VortexResult<Field> {
        Ok(Field::new(
            name,
            self.to_arrow_data_type(ext)?,
            ext.is_nullable(),
        ))
    }
}

/// Plugin trait that converts an Arrow [`Field`] into a Vortex [`DType`].
///
/// Readers are registered as a chain on [`crate::arrow::ArrowSession`] and walked in
/// registration order, with user-registered readers running before the built-in readers so
/// external crates can override built-in behavior.
///
/// Returning [`Ok(None)`] passes the request to the next reader in the chain.
pub trait ArrowDTypeReader: 'static + Send + Sync + Debug {
    /// Try to read a Vortex [`DType`] from an Arrow [`Field`].
    ///
    /// Implementations typically inspect [`Field::metadata`] for the `ARROW:extension:name`
    /// key and dispatch on it.
    fn try_read_dtype(&self, field: &Field) -> VortexResult<Option<DType>>;
}
