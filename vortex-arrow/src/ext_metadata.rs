// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The Arrow field metadata that holds a Vortex extension type without an Arrow plugin.
//!
//! Arrow field metadata must be UTF-8, but Vortex metadata is arbitrary bytes, so the metadata is
//! always base64, and [`EXTENSION_METADATA_ENCODING_KEY`] marks it as base64. The marker is needed
//! because other producers write the same Arrow extension keys with plain text metadata.

use arrow_schema::Field;
use arrow_schema::extension::EXTENSION_TYPE_METADATA_KEY;
use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

/// The Arrow field metadata key that names the encoding of `ARROW:extension:metadata`.
///
/// The only encoding is `base64`. Without this key, the metadata is its own UTF-8 bytes.
pub const EXTENSION_METADATA_ENCODING_KEY: &str = "vortex:extension:metadata:encoding";

/// The value of [`EXTENSION_METADATA_ENCODING_KEY`] for standard padded base64.
const BASE64_ENCODING: &str = "base64";

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
        Some(BASE64_ENCODING) => BASE64_STANDARD.decode(metadata).map_err(|err| {
            vortex_err!(
                "Arrow field {:?} must carry base64 extension metadata, got {metadata:?}: {err}",
                field.name()
            )
        }),
        Some(encoding) => vortex_bail!(
            "Arrow field {:?} must have the extension metadata encoding {BASE64_ENCODING:?}, got {encoding:?}",
            field.name()
        ),
    }
}
