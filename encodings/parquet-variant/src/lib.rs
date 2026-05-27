// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex support for Arrow's canonical `arrow.parquet.variant` extension type.
//!
//! This encoding provides a lossless representation of semi-structured data stored as
//! [Parquet Variant values] inside Arrow columns. It also supports [shredded variant values],
//! allowing systems to pass variant-encoded data around without special handling unless they
//! need to inspect the encoded contents directly.
//!
//! The storage type is a `Struct` that follows the Arrow canonical extension contract:
//! - `metadata` (required): a non-nullable binary child containing variant metadata
//! - `value` (optional): a binary child containing unshredded variant bytes
//! - `typed_value` (optional): a shredded child with a primitive, list, or struct layout
//!
//! At least one of `value` or `typed_value` must be present. Nested shredded values recurse
//! through the same `value` and `typed_value` structure described by the canonical extension
//! type documentation.
//!
//! See the Arrow canonical extension docs for the storage rules, and the Parquet format
//! specification for the binary representation.
//!
//! [Parquet Variant values]: https://github.com/apache/parquet-format/blob/master/VariantEncoding.md
//! [shredded variant values]: https://github.com/apache/parquet-format/blob/master/VariantShredding.md
//! [Arrow canonical extension type]: https://arrow.apache.org/docs/format/CanonicalExtensions.html#parquet-variant

mod array;
mod arrow;
mod kernel;
mod operations;
mod validity;
mod vtable;

use std::sync::Arc;

pub use array::ParquetVariantArrayExt;
pub use arrow::PARQUET_VARIANT_ARROW_EXTENSION_NAME;
use vortex_array::arrow::ArrowSessionExt;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;
pub use vtable::ParquetVariant;
pub use vtable::ParquetVariantArray;

/// Register Parquet Variant array and Arrow extension support with a session.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(ParquetVariant);
    session.arrow().register_exporter(Arc::new(ParquetVariant));
    session.arrow().register_importer(Arc::new(ParquetVariant));
}

#[cfg(test)]
mod arrow_session_tests {
    use std::sync::Arc;

    use arrow_array::Array as _;
    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_array::StructArray;
    use arrow_array::cast::AsArray;
    use arrow_schema::Field;
    use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
    use parquet_variant::Variant as PqVariant;
    use parquet_variant_compute::VariantArrayBuilder;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrow::ArrowSessionExt;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ParquetVariant;

    fn session() -> VortexSession {
        let session = VortexSession::empty().with::<ArraySession>();
        crate::initialize(&session);
        session
    }

    fn arrow_variant_storage() -> StructArray {
        let mut builder = VariantArrayBuilder::new(3);
        builder.append_variant(PqVariant::from(42i8));
        builder.append_variant(PqVariant::from(true));
        builder.append_variant(PqVariant::from("vortex"));
        builder.build().into_inner()
    }

    fn arrow_variant_field(storage: &StructArray) -> Field {
        Field::new("variant", storage.data_type().clone(), false).with_metadata(
            [(
                EXTENSION_TYPE_NAME_KEY.to_string(),
                "arrow.parquet.variant".to_string(),
            )]
            .into(),
        )
    }

    #[test]
    fn arrow_session_imports_parquet_variant_extension_array() -> VortexResult<()> {
        let session = session();
        let storage = arrow_variant_storage();
        let field = arrow_variant_field(&storage);
        let imported = session
            .arrow()
            .from_arrow_array(Arc::new(storage) as ArrowArrayRef, &field)?;

        assert_eq!(imported.dtype(), &DType::Variant(Nullability::NonNullable));
        assert!(imported.as_opt::<ParquetVariant>().is_some());
        Ok(())
    }

    #[test]
    fn arrow_session_exports_parquet_variant_extension_array() -> VortexResult<()> {
        let session = session();
        let storage = arrow_variant_storage();
        let field = arrow_variant_field(&storage);
        let imported = session
            .arrow()
            .from_arrow_array(Arc::new(storage.clone()) as ArrowArrayRef, &field)?;

        let mut ctx = session.create_execution_ctx();
        let exported = session
            .arrow()
            .execute_arrow(imported, Some(&field), &mut ctx)?;
        let exported = exported.as_struct();

        assert_eq!(exported.len(), storage.len());
        assert_eq!(exported.column_names(), storage.column_names());
        assert_eq!(exported.fields(), storage.fields());
        for (actual, expected) in exported.columns().iter().zip(storage.columns()) {
            assert_eq!(actual.to_data(), expected.to_data());
        }
        Ok(())
    }
}
