// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// TODO(connor): We don't use the `TemporalArray` right now because it doesn't have easy interop yet
// for the chrono `DateTime` type, and bringing in arrow for just this is too heavyweight.

use std::sync::Arc;

use serde::Deserialize;
use serde::Serialize;
use vortex::dtype::DType;
use vortex::dtype::FieldNames;
use vortex::dtype::Nullability::NonNullable;
use vortex::dtype::PType;
use vortex::dtype::StructFields;
use vortex::scalar::Scalar;
use vortex_array::Array;
use vortex_array::ToCanonical;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::website::commit_id::CommitId;

/// Commit information including author, message, timestamp, and commit ID.
///
/// The field order determines the derived [`Ord`] implementation: timestamp first, then author,
/// message, and finally commit_id.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CommitInfo {
    /// Unix timestamp in seconds.
    timestamp: i64,
    author: Author,
    message: String,
    commit_id: CommitId,
}

impl CommitInfo {
    /// Creates a new [`CommitInfo`].
    pub fn new(timestamp: i64, author: Author, message: String, commit_id: CommitId) -> Self {
        Self {
            timestamp,
            author,
            message,
            commit_id,
        }
    }

    /// Returns the commit timestamp as a Unix timestamp in seconds.
    pub fn timestamp(&self) -> i64 {
        self.timestamp
    }

    /// Returns the commit author.
    pub fn author(&self) -> &Author {
        &self.author
    }

    /// Returns the commit message.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Returns the commit ID.
    pub fn commit_id(&self) -> &CommitId {
        &self.commit_id
    }

    /// Returns the [`DType`] for a [`CommitInfo`].
    ///
    /// The schema is:
    /// - `timestamp`: `i64` (Unix timestamp in seconds)
    /// - `author`: Struct (name: Utf8, email: Utf8)
    /// - `message`: `Utf8`
    /// - `commit_id`: `FixedSizeList<u8, 20>` (20-byte binary SHA-1)
    pub fn dtype() -> DType {
        DType::Struct(
            StructFields::new(
                FieldNames::from(["timestamp", "author", "message", "commit_id"]),
                vec![
                    DType::Primitive(PType::I64, NonNullable),
                    Author::dtype(),
                    DType::Utf8(NonNullable),
                    DType::FixedSizeList(
                        Arc::new(DType::Primitive(PType::U8, NonNullable)),
                        20,
                        NonNullable,
                    ),
                ],
            ),
            NonNullable,
        )
    }

    /// Converts a [`CommitInfo`] to a [`Scalar`].
    pub fn into_scalar(&self) -> Scalar {
        let u8_dtype = DType::Primitive(PType::U8, NonNullable);

        // Convert the 20-byte commit_id to a FixedSizeList scalar.
        let commit_id_bytes: Vec<Scalar> = self
            .commit_id
            .0
            .iter()
            .map(|&b| Scalar::primitive(b, NonNullable))
            .collect();
        let commit_id_scalar = Scalar::fixed_size_list(u8_dtype, commit_id_bytes, NonNullable);

        Scalar::struct_(
            Self::dtype(),
            vec![
                Scalar::primitive(self.timestamp, NonNullable),
                self.author.into_scalar(),
                Scalar::utf8(self.message.as_str(), NonNullable),
                commit_id_scalar,
            ],
        )
    }

    /// Converts a Vortex array (expected to be a struct array) into a vector of [`CommitInfo`].
    ///
    /// The array must have the following schema:
    /// - `timestamp`: `i64`
    /// - `author`: Struct (name: Utf8, email: Utf8)
    /// - `message`: `Utf8`
    /// - `commit_id`: `FixedSizeList<u8, 20>`
    pub fn vec_from_array(array: &dyn Array) -> VortexResult<Vec<Self>> {
        let struct_array: StructArray = array.to_struct();
        let len = struct_array.len();
        let mut entries = Vec::with_capacity(len);

        // Extract each field.
        let timestamp_field = struct_array.field_by_name("timestamp")?;
        let author_field = struct_array.field_by_name("author")?;
        let message_field = struct_array.field_by_name("message")?;
        let commit_id_field = struct_array.field_by_name("commit_id")?;

        // Convert timestamp to primitive array.
        let timestamp_prim: PrimitiveArray = timestamp_field.to_primitive();
        let timestamps: &[i64] = timestamp_prim.as_slice();

        // Convert author struct to its components.
        let author_struct: StructArray = author_field.to_struct();
        let author_name_field = author_struct.field_by_name("name")?;
        let author_email_field = author_struct.field_by_name("email")?;
        let author_name_vbv = author_name_field.to_varbinview();
        let author_email_vbv = author_email_field.to_varbinview();

        // Convert message to varbinview.
        let message_vbv = message_field.to_varbinview();

        // Convert commit_id to canonical fixed-size list and get the underlying bytes.
        let commit_id_fsl: FixedSizeListArray = commit_id_field.to_fixed_size_list();
        if commit_id_fsl.list_size() != 20 {
            vortex_bail!(
                "Expected commit_id to have list_size 20, got {}",
                commit_id_fsl.list_size()
            );
        }
        let commit_id_elements: PrimitiveArray = commit_id_fsl.elements().to_primitive();
        let commit_id_bytes: &[u8] = commit_id_elements.as_slice();

        // Build the entries.
        for i in 0..len {
            // Extract author fields.
            let name = std::str::from_utf8(author_name_vbv.bytes_at(i).as_ref())
                .map_err(|e| vortex_error::vortex_err!("Invalid UTF-8 in author name: {}", e))?
                .to_string();
            let email = std::str::from_utf8(author_email_vbv.bytes_at(i).as_ref())
                .map_err(|e| vortex_error::vortex_err!("Invalid UTF-8 in author email: {}", e))?
                .to_string();

            // Extract message.
            let message = std::str::from_utf8(message_vbv.bytes_at(i).as_ref())
                .map_err(|e| vortex_error::vortex_err!("Invalid UTF-8 in message: {}", e))?
                .to_string();

            // Extract the 20-byte commit_id for this row.
            let start = i * 20;
            let end = start + 20;
            let mut commit_id_arr = [0u8; 20];
            commit_id_arr.copy_from_slice(&commit_id_bytes[start..end]);

            entries.push(CommitInfo {
                timestamp: timestamps[i],
                author: Author::new(name, email),
                message,
                commit_id: CommitId(commit_id_arr),
            });
        }

        Ok(entries)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Author {
    name: String,
    email: String,
}

impl Author {
    /// Creates a new [`Author`].
    pub fn new(name: String, email: String) -> Self {
        Self { name, email }
    }

    /// Returns the author's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the author's email.
    pub fn email(&self) -> &str {
        &self.email
    }

    /// Returns the [`DType`] for an [`Author`].
    ///
    /// The schema is:
    /// - `name`: `Utf8`
    /// - `email`: `Utf8`
    pub fn dtype() -> DType {
        DType::Struct(
            StructFields::new(
                FieldNames::from(["name", "email"]),
                vec![DType::Utf8(NonNullable), DType::Utf8(NonNullable)],
            ),
            NonNullable,
        )
    }

    /// Converts an [`Author`] to a [`Scalar`].
    pub fn into_scalar(&self) -> Scalar {
        Scalar::struct_(
            Self::dtype(),
            vec![
                Scalar::utf8(self.name.as_str(), NonNullable),
                Scalar::utf8(self.email.as_str(), NonNullable),
            ],
        )
    }
}
