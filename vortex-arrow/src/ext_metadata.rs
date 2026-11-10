// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Encoding Vortex extension types in Arrow field metadata.
//!
//! Typed plugins use [`crate::ArrowSession::register_extension`], whose adapter handles Vortex
//! metadata without exposing its keys or encoding to the callbacks.
//!
//! A custom extension with no Arrow [`ExtensionType`] implementation can use
//! [`set_arrow_extension_name`] to name its storage field. An extension without an Arrow plugin
//! uses [`ArrowSession::to_arrow_storage_field`] instead.
//!
//! Field metadata must be UTF-8, so Vortex payloads use marked base64. The Vortex ID path also
//! accepts unmarked UTF-8 metadata from older producers. A separate Vortex payload requires the
//! marker, and an absent payload imposes no constraint on a target's Vortex metadata.
//!
//! [`ArrowSession::to_arrow_storage_field`]: crate::ArrowSession::to_arrow_storage_field
//! [`ExtensionType`]: arrow_schema::extension::ExtensionType

use arrow_schema::Field;
use arrow_schema::extension::EXTENSION_TYPE_METADATA_KEY;
use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDTypePlugin;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::dtype::extension::ExtVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

/// The Arrow field metadata key that names the encoding of a Vortex metadata payload.
///
/// This applies to `ARROW:extension:metadata` and [`VORTEX_EXTENSION_METADATA_KEY`]. The only
/// encoding is `base64`. Without this key, `ARROW:extension:metadata` is read as raw UTF-8 for
/// backward compatibility.
pub const EXTENSION_METADATA_ENCODING_KEY: &str = "vortex:extension:metadata:encoding";

/// Vortex metadata retained alongside an Arrow extension type.
///
/// The payload is base64 and **must** carry [`EXTENSION_METADATA_ENCODING_KEY`]. It is separate
/// from the Arrow extension's own metadata, whose format is defined by that extension.
pub const VORTEX_EXTENSION_METADATA_KEY: &str = "vortex:extension:metadata";

/// The value of [`EXTENSION_METADATA_ENCODING_KEY`] for standard padded base64.
const BASE64_ENCODING: &str = "base64";

/// Sets an Arrow extension name while preserving the field's other metadata.
///
/// Prefer [`Field::try_with_extension_type`] when an Arrow extension has an
/// [`ExtensionType`](arrow_schema::extension::ExtensionType) implementation, which also validates
/// its storage and native metadata.
pub fn set_arrow_extension_name(field: &mut Field, name: impl Into<String>) {
    field
        .metadata_mut()
        .insert(EXTENSION_TYPE_NAME_KEY.to_owned(), name.into());
}

/// Writes Vortex metadata alongside an Arrow field's native metadata.
///
/// Serialization, keys, and encoding are owned by this module. Callers decide whether the Arrow
/// representation needs a separate Vortex payload. A failed serialization leaves `field` unchanged.
pub(crate) fn write_vortex_metadata<V: ExtVTable>(
    field: &mut Field,
    vtable: &V,
    metadata: &V::Metadata,
) -> VortexResult<()> {
    let payload = BASE64_STANDARD.encode(vtable.serialize_metadata(metadata)?);
    field.metadata_mut().extend([
        (VORTEX_EXTENSION_METADATA_KEY.to_owned(), payload),
        (
            EXTENSION_METADATA_ENCODING_KEY.to_owned(),
            BASE64_ENCODING.to_owned(),
        ),
    ]);

    Ok(())
}

/// Restores a Vortex dtype using metadata and the importer's inferred Vortex storage.
///
/// Returns `None` when the field has no Vortex payload. A present payload **must** have a
/// supported encoding marker and metadata compatible with `storage_dtype`. Native Arrow metadata
/// is not read. Reconstruction goes through the dtype plugin so its storage checks also apply.
pub(crate) fn read_vortex_metadata<V: ExtVTable>(
    field: &Field,
    vtable: &V,
    storage_dtype: &DType,
) -> VortexResult<Option<ExtDTypeRef>> {
    let Some(payload) = field.metadata().get(VORTEX_EXTENSION_METADATA_KEY) else {
        return Ok(None);
    };

    vortex_ensure_eq!(
        field
            .metadata()
            .get(EXTENSION_METADATA_ENCODING_KEY)
            .map(String::as_str)
            .unwrap_or_default(),
        BASE64_ENCODING,
        "Arrow field {:?} must mark its Vortex metadata as base64",
        field.name()
    );
    let metadata = decode_base64_metadata(field, payload)?;

    vtable
        .deserialize(&metadata, storage_dtype.clone())
        .map(Some)
        .map_err(|err| {
            err.with_context(format!(
                "Invalid Vortex metadata on Arrow field {:?}",
                field.name()
            ))
        })
}

/// Returns the Arrow field metadata that holds `ext_dtype`.
pub fn encode_arrow_ext_metadata(ext_dtype: &ExtDTypeRef) -> VortexResult<[(String, String); 3]> {
    let metadata = BASE64_STANDARD.encode(ext_dtype.serialize_metadata()?);

    Ok([
        (
            EXTENSION_TYPE_NAME_KEY.to_owned(),
            ext_dtype.id().to_string(),
        ),
        (EXTENSION_TYPE_METADATA_KEY.to_owned(), metadata),
        (
            EXTENSION_METADATA_ENCODING_KEY.to_owned(),
            BASE64_ENCODING.to_owned(),
        ),
    ])
}

/// Returns the extension metadata bytes that `field` holds, which are empty if it has none.
///
/// An unmarked value is its own UTF-8 bytes. Fails if the encoding is not `base64`, or if the value
/// is not valid base64.
pub fn decode_arrow_ext_metadata(field: &Field) -> VortexResult<Vec<u8>> {
    let Some(metadata) = field.extension_type_metadata() else {
        return Ok(Vec::new());
    };

    match field
        .metadata()
        .get(EXTENSION_METADATA_ENCODING_KEY)
        .map(String::as_str)
    {
        None => Ok(metadata.as_bytes().to_vec()),
        Some(BASE64_ENCODING) => decode_base64_metadata(field, metadata),
        Some(encoding) => vortex_bail!(
            "Arrow field {:?} must have the extension metadata encoding {BASE64_ENCODING:?}, got {encoding:?}",
            field.name()
        ),
    }
}

fn decode_base64_metadata(field: &Field, metadata: &str) -> VortexResult<Vec<u8>> {
    BASE64_STANDARD.decode(metadata).map_err(|err| {
        vortex_err!(
            "Arrow field {:?} must carry base64 extension metadata, got {metadata:?}: {err}",
            field.name()
        )
    })
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::extension::EXTENSION_TYPE_METADATA_KEY;
    use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
    use rstest::rstest;
    use vortex_array::dtype::PType;
    use vortex_array::extension::datetime::Date;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_error::VortexResult;

    use super::*;

    #[track_caller]
    fn check_metadata_round_trip(metadata: TimeUnit, native_type: DataType) -> VortexResult<()> {
        let mut field = Field::new("date", native_type, true);
        let inferred = Date::try_new(metadata, true.into())?.erased();
        field.metadata_mut().extend([
            (
                EXTENSION_TYPE_NAME_KEY.to_owned(),
                "example.date".to_owned(),
            ),
            (
                EXTENSION_TYPE_METADATA_KEY.to_owned(),
                "native metadata".to_owned(),
            ),
            ("application".to_owned(), "retained".to_owned()), // Unrelated application metadata.
        ]);
        let original = field.clone();

        write_vortex_metadata(&mut field, &Date, &metadata)?;

        assert_eq!(
            read_vortex_metadata(&field, &Date, inferred.storage_dtype())?,
            Some(inferred)
        );
        assert_eq!(field.data_type(), original.data_type());
        assert_eq!(field.name(), original.name());
        assert_eq!(field.is_nullable(), original.is_nullable());
        for (key, value) in original.metadata() {
            assert_eq!(field.metadata().get(key), Some(value));
        }

        Ok(())
    }

    #[rstest]
    #[case::days(TimeUnit::Days, DataType::Date32)]
    #[case::milliseconds(TimeUnit::Milliseconds, DataType::Date64)]
    fn typed_metadata_preserves_native_field(
        #[case] metadata: TimeUnit,
        #[case] native_type: DataType,
    ) -> VortexResult<()> {
        check_metadata_round_trip(metadata, native_type)
    }

    #[test]
    fn absent_vortex_metadata_ignores_native_metadata() -> VortexResult<()> {
        let mut field = Field::new("date", DataType::Date32, false);
        field.metadata_mut().extend([
            (
                EXTENSION_TYPE_METADATA_KEY.to_owned(),
                "native metadata".to_owned(),
            ),
            (
                EXTENSION_METADATA_ENCODING_KEY.to_owned(),
                "native encoding".to_owned(),
            ),
        ]);

        assert_eq!(
            read_vortex_metadata(&field, &Date, &DType::Primitive(PType::I32, false.into()))?,
            None
        );

        Ok(())
    }

    #[rstest]
    #[case::missing_encoding("AA==", None)]
    #[case::unsupported_encoding("AA==", Some("hex"))]
    #[case::invalid_base64("!", Some("base64"))]
    #[case::invalid_typed_metadata("/w==", Some("base64"))]
    #[case::empty_typed_metadata("", Some("base64"))]
    fn malformed_metadata_is_rejected(#[case] payload: &str, #[case] encoding: Option<&str>) {
        let mut field = Field::new("date", DataType::Date32, false);
        field
            .metadata_mut()
            .insert(VORTEX_EXTENSION_METADATA_KEY.to_owned(), payload.to_owned());
        if let Some(encoding) = encoding {
            field.metadata_mut().insert(
                EXTENSION_METADATA_ENCODING_KEY.to_owned(),
                encoding.to_owned(),
            );
        }

        let error =
            read_vortex_metadata(&field, &Date, &DType::Primitive(PType::I32, false.into()))
                .expect_err("the metadata is malformed");
        assert!(error.to_string().contains("date"));
    }

    #[test]
    fn metadata_must_match_inferred_storage() -> VortexResult<()> {
        let mut field = Field::new("date", DataType::Date32, false);
        write_vortex_metadata(&mut field, &Date, &TimeUnit::Milliseconds)?;

        assert!(
            read_vortex_metadata(&field, &Date, &DType::Primitive(PType::I32, false.into()))
                .is_err()
        );

        Ok(())
    }

    #[test]
    fn extension_name_preserves_other_metadata() {
        let mut field = Field::new("value", DataType::Float32, false);
        field
            .metadata_mut()
            .insert("application".to_owned(), "retained".to_owned());

        set_arrow_extension_name(&mut field, "example.vector");

        assert_eq!(field.extension_type_name(), Some("example.vector"));
        assert_eq!(
            field.metadata().get("application").map(String::as_str),
            Some("retained")
        );
    }
}
