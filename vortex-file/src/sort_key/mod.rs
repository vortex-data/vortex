// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sort keys for Vortex files.
//!
//! The current implementations in this file are for validating sort keys,
//! and for writing and reading sort keys to/from the footer.
//!
//! # About sort keys
//!
//! Sort keys are a compound key that describes how the rows in a file are sorted.
//! A vortex file can store one or more sorting keys, each of which must hold
//! for the same rows. For example, these rows are lexicographically sorted by
//! both (user_id, city) and (timestamp, user_id):
//!
//! ```text
//! user_id | city       | timestamp
//! --------+------------+----------
//! 1       | Bangkok    | 10
//! 1       | Mumbai     | 20
//! 2       | Tokyo      | 20
//! 3       | Copenhagen | 30
//! ```
//!
//! The first sort key orders cities within each user. The second key orders user IDs
//! within each timestamp.
//!
//! # Difference from sortedness stats
//!
//! The difference between `IsSorted` / `IsStrictSorted` and sort keys
//! is that the stats describe the sortedness of one column, while a compound key
//! requires sortedness across the file only for its first column. The remaining
//! columns only need to be sorted where all preceding columns are equal.
//!
//! The ordering must also hold across chunks. For example, each chunk below
//! is sorted by (user_id, city):
//!
//! ```text
//!          user_id | city
//!          --------+-----------
//! Chunk 1        1 | Bangkok
//!                2 | Tokyo
//!          --------+-----------
//! Chunk 2        2 | Mumbai      <- breaks the ordering
//!                3 | Copenhagen
//! ```
//!
//! But together, they break the sort key: (2, Tokyo) comes before (2, Mumbai),
//! even though Mumbai should come first.

use std::sync::Arc;

use flatbuffers::FlatBufferBuilder;
use flatbuffers::WIPOffset;
use vortex_array::dtype::DType;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldPath;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_set::HashSet;

use crate::flatbuffers::footer as fb;

/// Represents a compound sort key, with fields listed in comparison order.
pub type SortKey = [FieldPath];
pub type SortKeyRef = Arc<SortKey>;

/// Check that the key is non-empty and its fields exist and appear only once.
pub(crate) fn validate_sort_key(dtype: &DType, fields: &SortKey) -> VortexResult<()> {
    vortex_ensure!(
        !fields.is_empty(),
        "A file sort key must contain at least one field"
    );

    let mut seen = HashSet::<&FieldPath>::default();
    for field in fields {
        vortex_ensure!(seen.insert(field), "Duplicate sort field: {}", field);

        match (dtype, field.parts()) {
            (DType::Struct(struct_fields, _), [Field::Name(name)]) => {
                vortex_ensure!(
                    struct_fields.find(name).is_some(),
                    "Sort field {} does not exist in the file schema",
                    field
                );
            }
            // For a non-struct file, an empty path selects the entire root array as the sort field.
            (dtype, []) if !dtype.is_struct() => {}
            _ => vortex_bail!(
                "Sort field {} must select a top-level field of a struct, or a non-struct root",
                field
            ),
        }
    }

    Ok(())
}

/// Reads and validates the footer's sort key.
pub(crate) fn read_sort_key(key: fb::SortKeySpec<'_>, dtype: &DType) -> VortexResult<SortKeyRef> {
    let fields: SortKeyRef = key
        .fields()
        .iter()
        .map(|field| field.path().iter().map(Field::from).collect())
        .collect();
    validate_sort_key(dtype, &fields)?;

    Ok(fields)
}

/// Writes the sort key into the footer.
pub(crate) fn write_sort_key<'fb>(
    fbb: &mut FlatBufferBuilder<'fb>,
    fields: &SortKey,
) -> VortexResult<WIPOffset<fb::SortKeySpec<'fb>>> {
    let mut offsets = Vec::with_capacity(fields.len());

    for field in fields {
        let names = field
            .parts()
            .iter()
            .map(|part| {
                let name = part
                    .as_name()
                    .ok_or_else(|| vortex_err!("Invalid sort field path"))?;
                Ok(fbb.create_string(name))
            })
            .collect::<VortexResult<Vec<_>>>()?;

        let path = fbb.create_vector(&names);
        offsets.push(fb::FieldPath::create(
            fbb,
            &fb::FieldPathArgs { path: Some(path) },
        ));
    }

    let fields = fbb.create_vector(&offsets);
    Ok(fb::SortKeySpec::create(
        fbb,
        &fb::SortKeySpecArgs {
            fields: Some(fields),
        },
    ))
}
