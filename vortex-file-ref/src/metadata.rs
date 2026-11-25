// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The fields that a file reference column defines.

use std::fmt;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

/// One field of a file reference.
///
/// The names and meanings follow the Parquet FILE logical type. Every field is nullable, and a
/// column defines only the fields it uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FileRefField {
    /// An RFC 3986 URI reference, absolute or relative, that locates the external bytes.
    Uri,
    /// The start of the byte range within the file at [`Self::Uri`]. Unset means zero.
    Offset,
    /// The length of the byte range. Unset with a [`Self::Uri`] means the whole file.
    Size,
    /// The MIME type of the referenced bytes. Unset means `application/octet-stream`.
    ContentType,
    /// An `<algorithm>:<digest>` integrity token for the referenced bytes.
    Checksum,
    /// The referenced bytes, stored in the value itself.
    Inline,
}

impl FileRefField {
    /// Every field, in the order that the Parquet FILE logical type lists them.
    pub const ALL: [Self; 6] = [
        Self::Uri,
        Self::Offset,
        Self::Size,
        Self::ContentType,
        Self::Checksum,
        Self::Inline,
    ];

    /// The case-sensitive name of the field.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Uri => "uri",
            Self::Offset => "offset",
            Self::Size => "size",
            Self::ContentType => "content_type",
            Self::Checksum => "checksum",
            Self::Inline => "inline",
        }
    }

    /// Looks up a field by its case-sensitive name.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|field| field.name() == name)
    }

    /// The storage dtype of the field, which is always nullable.
    pub fn dtype(self) -> DType {
        match self {
            Self::Uri | Self::ContentType | Self::Checksum => DType::Utf8(Nullability::Nullable),
            Self::Offset | Self::Size => DType::Primitive(PType::I64, Nullability::Nullable),
            Self::Inline => DType::Binary(Nullability::Nullable),
        }
    }
}

impl fmt::Display for FileRefField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The fields that a file reference column defines, in storage order.
///
/// A column defines at least one field and each field at most once, in any order. The order is
/// kept so that a column exports to the same Arrow struct that it imports from.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FileRefMetadata {
    /// The defined fields, in storage order.
    fields: Arc<[FileRefField]>,
}

impl FileRefMetadata {
    /// Creates metadata for the given fields, in storage order.
    ///
    /// # Errors
    ///
    /// Returns an error if `fields` is empty or names a field more than once.
    pub fn try_new(fields: impl IntoIterator<Item = FileRefField>) -> VortexResult<Self> {
        let metadata = Self {
            fields: fields.into_iter().collect(),
        };
        metadata.validate()?;
        Ok(metadata)
    }

    /// Metadata that defines every field, in the order that the Parquet FILE logical type lists
    /// them.
    pub fn all() -> Self {
        Self {
            fields: FileRefField::ALL.into(),
        }
    }

    /// The defined fields, in storage order.
    pub fn fields(&self) -> &[FileRefField] {
        &self.fields
    }

    /// The storage position of `field`, if the column defines it.
    pub fn position(&self, field: FileRefField) -> Option<usize> {
        self.fields.iter().position(|defined| *defined == field)
    }

    /// Checks that at least one field is defined and none is defined twice.
    pub(crate) fn validate(&self) -> VortexResult<()> {
        vortex_ensure!(
            !self.fields.is_empty(),
            "a file reference must define at least one field"
        );
        for (index, field) in self.fields.iter().enumerate() {
            vortex_ensure!(
                !self.fields[..index].contains(field),
                "a file reference defines field {field} more than once"
            );
        }
        Ok(())
    }

    /// Reads the defined fields from the names of a struct storage dtype.
    ///
    /// Only the names are read. The dtype constructors compare the declared storage, including
    /// the field types and their nullability, with the supplied storage.
    pub(crate) fn from_storage(storage_dtype: &DType) -> VortexResult<Self> {
        let DType::Struct(fields, _) = storage_dtype else {
            vortex_bail!("file reference storage must be a struct, got {storage_dtype}");
        };
        Self::try_new(
            fields
                .names()
                .iter()
                .map(|name| {
                    FileRefField::from_name(name.as_ref())
                        .ok_or_else(|| vortex_err!("unknown file reference field {name:?}"))
                })
                .collect::<VortexResult<Vec<_>>>()?,
        )
    }

    /// The struct storage dtype with non-nullable rows.
    pub(crate) fn storage_dtype(&self) -> DType {
        let names = self
            .fields
            .iter()
            .map(|field| field.name())
            .collect::<FieldNames>();
        let dtypes = self.fields.iter().map(|field| field.dtype()).collect();
        DType::Struct(StructFields::new(names, dtypes), Nullability::NonNullable)
    }
}

impl fmt::Display for FileRefMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, field) in self.fields.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{field}")?;
        }
        Ok(())
    }
}
