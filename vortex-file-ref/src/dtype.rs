// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The file reference extension dtype.

use std::fmt;

use prost::Message;
use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::PValue;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;

use crate::FileRef;
use crate::FileRefField;
use crate::FileRefMetadata;

/// The serialized metadata of the file reference dtype.
///
/// The storage dtype records the defined fields, so the message has no fields yet. Decoding it as
/// a message lets a later writer add fields that are safe to ignore.
#[derive(Clone, PartialEq, Message)]
struct FileRefMetadataProto {}

/// A non-null file reference value.
///
/// Each field is `None` when the column does not define it or the value leaves it null.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FileRefValue<'a> {
    /// The URI reference that locates the external bytes.
    pub uri: Option<&'a str>,
    /// The start of the byte range within the file at `uri`.
    pub offset: Option<i64>,
    /// The length of the byte range.
    pub size: Option<i64>,
    /// The MIME type of the referenced bytes.
    pub content_type: Option<&'a str>,
    /// An `<algorithm>:<digest>` integrity token for the referenced bytes.
    pub checksum: Option<&'a str>,
    /// The referenced bytes, stored in the value itself.
    pub inline: Option<&'a [u8]>,
}

impl FileRefValue<'_> {
    /// Checks the value against the validation rules of the Parquet FILE logical type.
    ///
    /// A value must resolve to bytes, so it sets `inline` or `uri`. An `offset` requires both
    /// `uri` and `size`, and neither `offset` nor `size` may be negative. As the spec says, an empty
    /// string counts as unset.
    ///
    /// # Errors
    ///
    /// Returns an error if the value breaks one of these rules.
    pub fn validate(&self) -> VortexResult<()> {
        let has_uri = self.uri.is_some_and(|uri| !uri.is_empty());
        vortex_ensure!(
            has_uri || self.inline.is_some(),
            "a file reference must set uri or inline"
        );
        if let Some(offset) = self.offset {
            vortex_ensure!(has_uri, "a file reference offset requires a uri");
            vortex_ensure!(
                self.size.is_some(),
                "a file reference offset requires a size"
            );
            vortex_ensure!(
                offset >= 0,
                "a file reference offset must not be negative, got {offset}"
            );
        }
        if let Some(size) = self.size {
            vortex_ensure!(
                size >= 0,
                "a file reference size must not be negative, got {size}"
            );
        }
        Ok(())
    }
}

impl fmt::Display for FileRefValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("{")?;
        let mut separator = "";
        let mut field = |f: &mut fmt::Formatter<'_>, name: &str, value: &dyn fmt::Display| {
            let result = write!(f, "{separator}{name}: {value}");
            separator = ", ";
            result
        };
        if let Some(uri) = self.uri {
            field(f, "uri", &format_args!("{uri:?}"))?;
        }
        if let Some(offset) = self.offset {
            field(f, "offset", &offset)?;
        }
        if let Some(size) = self.size {
            field(f, "size", &size)?;
        }
        if let Some(content_type) = self.content_type {
            field(f, "content_type", &format_args!("{content_type:?}"))?;
        }
        if let Some(checksum) = self.checksum {
            field(f, "checksum", &format_args!("{checksum:?}"))?;
        }
        if let Some(inline) = self.inline {
            field(f, "inline", &format_args!("<{} bytes>", inline.len()))?;
        }
        f.write_str("}")
    }
}

impl ExtVTable for FileRef {
    type Metadata = FileRefMetadata;
    type StoredMetadata = ();
    type NativeValue<'a> = FileRefValue<'a>;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("vortex.file_ref");
        *ID
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(FileRefMetadataProto {}.encode_to_vec())
    }

    fn deserialize_metadata(&self, bytes: &[u8]) -> VortexResult<Self::StoredMetadata> {
        FileRefMetadataProto::decode(bytes)
            .map_err(|err| vortex_err!("invalid file reference metadata: {err}"))?;
        Ok(())
    }

    fn metadata_from_storage(
        &self,
        _stored_metadata: Self::StoredMetadata,
        storage_dtype: &DType,
    ) -> VortexResult<Self::Metadata> {
        FileRefMetadata::from_storage(storage_dtype)
    }

    fn storage_dtype(metadata: &Self::Metadata) -> DType {
        metadata.storage_dtype()
    }

    fn validate_metadata(metadata: &Self::Metadata) -> VortexResult<()> {
        metadata.validate()
    }

    fn unpack_native<'a>(
        ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        let ScalarValue::Tuple(values) = storage_value else {
            vortex_bail!("file reference storage scalar must be a struct, got {storage_value}");
        };
        let fields = ext_dtype.metadata().fields();
        vortex_ensure_eq!(
            values.len(),
            fields.len(),
            "file reference storage scalar field count mismatch"
        );

        let mut value = FileRefValue::default();
        for (field, field_value) in fields.iter().zip(values) {
            let Some(field_value) = field_value else {
                continue;
            };
            match field {
                FileRefField::Uri => value.uri = Some(utf8(*field, field_value)?),
                FileRefField::Offset => value.offset = Some(int64(*field, field_value)?),
                FileRefField::Size => value.size = Some(int64(*field, field_value)?),
                FileRefField::ContentType => value.content_type = Some(utf8(*field, field_value)?),
                FileRefField::Checksum => value.checksum = Some(utf8(*field, field_value)?),
                FileRefField::Inline => {
                    let ScalarValue::Binary(bytes) = field_value else {
                        vortex_bail!("file reference field inline must be binary, got {field_value}");
                    };
                    value.inline = Some(bytes.as_slice());
                }
            }
        }
        value.validate()?;
        Ok(value)
    }
}

/// Reads a string field of a file reference scalar.
fn utf8(field: FileRefField, value: &ScalarValue) -> VortexResult<&str> {
    let ScalarValue::Utf8(value) = value else {
        vortex_bail!("file reference field {field} must be utf8, got {value}");
    };
    Ok(value.as_str())
}

/// Reads an `i64` field of a file reference scalar.
fn int64(field: FileRefField, value: &ScalarValue) -> VortexResult<i64> {
    let ScalarValue::Primitive(PValue::I64(value)) = value else {
        vortex_bail!("file reference field {field} must be i64, got {value}");
    };
    Ok(*value)
}
