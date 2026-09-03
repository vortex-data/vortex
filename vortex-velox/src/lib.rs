// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]
#![expect(non_camel_case_types)]
#![forbid(clippy::todo)]
#![forbid(clippy::unimplemented)]

//! Native adapter contract between Vortex and Velox.

mod api;
mod array;
mod projection;
mod read_at;
mod schema;
mod source;
mod temporal;
mod visitor;

pub use api::*;
pub use array::vx_velox_array_export_arrow;
pub use array::vx_velox_array_get_field;
pub use array::vx_velox_array_invalid_count;
pub use array::vx_velox_arrow_memory_callbacks;
pub use projection::vx_velox_expression_select;
pub use projection::vx_velox_expression_select_with_row_index;
pub use read_at::vx_velox_buffer;
pub use read_at::vx_velox_read_at;
pub use read_at::vx_velox_read_at_callbacks;
pub use read_at::vx_velox_read_request;
pub use schema::vx_velox_source_export_schema;
pub use source::vx_velox_natural_split;
pub use source::vx_velox_source;
pub use source::vx_velox_source_prune_natural_splits;
pub use visitor::vx_velox_binary_view;
pub use visitor::vx_velox_bool_view;
pub use visitor::vx_velox_buffer_owner;
pub use visitor::vx_velox_byte_buffer_view;
pub use visitor::vx_velox_constant_view;
pub use visitor::vx_velox_dictionary_view;
pub use visitor::vx_velox_export_cursor;
pub use visitor::vx_velox_list_view;
pub use visitor::vx_velox_map_view;
pub use visitor::vx_velox_primitive_type;
pub use visitor::vx_velox_primitive_view;
pub use visitor::vx_velox_struct_view;
pub use visitor::vx_velox_validity_kind;
pub use visitor::vx_velox_varbin_kind;
pub use visitor::vx_velox_varbin_view;
pub use visitor::vx_velox_visit_request;
pub use visitor::vx_velox_visitor;

/// The current major version of the Vortex and Velox adapter ABI.
pub const VX_VELOX_ABI_VERSION: u32 = 5;

/// The adapter supports batched host range reads.
pub const VX_VELOX_CAPABILITY_BATCH_READ: u64 = 1 << 0;

/// The adapter can open callback-backed Vortex sources.
pub const VX_VELOX_CAPABILITY_CALLBACK_SOURCE: u64 = 1 << 1;

/// The adapter reports stable natural row splits.
pub const VX_VELOX_CAPABILITY_NATURAL_SPLITS: u64 = 1 << 2;

/// The adapter can visit canonical primitive values in retained blocks.
pub const VX_VELOX_CAPABILITY_PRIMITIVE_VISITOR: u64 = 1 << 3;

/// The adapter can export source schemas through the Arrow C Data Interface.
pub const VX_VELOX_CAPABILITY_ARROW_SCHEMA: u64 = 1 << 4;

/// The adapter can export one Vortex array through the Arrow C Data Interface.
pub const VX_VELOX_CAPABILITY_ARRAY_ARROW_EXPORT: u64 = 1 << 5;

/// The adapter can project absolute file-row indexes with scan fields.
pub const VX_VELOX_CAPABILITY_ROW_INDEX_PROJECTION: u64 = 1 << 6;

/// The adapter can prove that natural splits cannot match an expression.
pub const VX_VELOX_CAPABILITY_NATURAL_SPLIT_PRUNING: u64 = 1 << 7;

/// The callback reader observes host cancellation before each host read callback.
///
/// This capability does not claim cancellation during cached scans or CPU execution.
pub const VX_VELOX_CAPABILITY_READ_CANCELLATION: u64 = 1 << 8;

/// The adapter retains one prepared array across several Velox output windows.
pub const VX_VELOX_CAPABILITY_EXPORT_CURSOR: u64 = 1 << 9;

/// The adapter can omit row-index projection from scans with contiguous rows.
pub const VX_VELOX_CAPABILITY_PLAIN_PROJECTION: u64 = 1 << 10;

/// The adapter can visit canonical UTF-8 and binary values in retained blocks.
pub const VX_VELOX_CAPABILITY_VARBIN_VISITOR: u64 = 1 << 11;

/// The adapter can preserve dictionary arrays during native export.
pub const VX_VELOX_CAPABILITY_DICTIONARY_VISITOR: u64 = 1 << 12;

/// The adapter can preserve constant arrays during native export.
pub const VX_VELOX_CAPABILITY_CONSTANT_VISITOR: u64 = 1 << 13;

/// The adapter can visit canonical packed Boolean values in retained blocks.
pub const VX_VELOX_CAPABILITY_BOOL_VISITOR: u64 = 1 << 14;
/// The adapter can export Vortex day-based dates through the primitive visitor.
pub const VX_VELOX_CAPABILITY_DATE_VISITOR: u64 = 1 << 15;
/// The adapter can normalize Vortex decimals for the primitive visitor.
pub const VX_VELOX_CAPABILITY_DECIMAL_VISITOR: u64 = 1 << 16;
/// The adapter can preserve canonical struct children during native export.
pub const VX_VELOX_CAPABILITY_STRUCT_VISITOR: u64 = 1 << 17;
/// The adapter can preserve canonical list children during native export.
pub const VX_VELOX_CAPABILITY_LIST_VISITOR: u64 = 1 << 18;
/// The adapter can preserve canonical map children during native export.
pub const VX_VELOX_CAPABILITY_MAP_VISITOR: u64 = 1 << 19;

/// Return the adapter ABI version.
#[unsafe(no_mangle)]
pub extern "C" fn vx_velox_abi_version() -> u32 {
    VX_VELOX_ABI_VERSION
}

/// Return the capabilities implemented by this adapter build.
#[unsafe(no_mangle)]
pub extern "C" fn vx_velox_capabilities() -> u64 {
    VX_VELOX_CAPABILITY_BATCH_READ
        | VX_VELOX_CAPABILITY_CALLBACK_SOURCE
        | VX_VELOX_CAPABILITY_NATURAL_SPLITS
        | VX_VELOX_CAPABILITY_PRIMITIVE_VISITOR
        | VX_VELOX_CAPABILITY_ARROW_SCHEMA
        | VX_VELOX_CAPABILITY_ARRAY_ARROW_EXPORT
        | VX_VELOX_CAPABILITY_ROW_INDEX_PROJECTION
        | VX_VELOX_CAPABILITY_NATURAL_SPLIT_PRUNING
        | VX_VELOX_CAPABILITY_READ_CANCELLATION
        | VX_VELOX_CAPABILITY_EXPORT_CURSOR
        | VX_VELOX_CAPABILITY_PLAIN_PROJECTION
        | VX_VELOX_CAPABILITY_VARBIN_VISITOR
        | VX_VELOX_CAPABILITY_DICTIONARY_VISITOR
        | VX_VELOX_CAPABILITY_CONSTANT_VISITOR
        | VX_VELOX_CAPABILITY_BOOL_VISITOR
        | VX_VELOX_CAPABILITY_DATE_VISITOR
        | VX_VELOX_CAPABILITY_DECIMAL_VISITOR
        | VX_VELOX_CAPABILITY_STRUCT_VISITOR
        | VX_VELOX_CAPABILITY_LIST_VISITOR
        | VX_VELOX_CAPABILITY_MAP_VISITOR
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_contract() {
        assert_eq!(vx_velox_abi_version(), VX_VELOX_ABI_VERSION);
        assert_eq!(
            vx_velox_capabilities() & VX_VELOX_CAPABILITY_BATCH_READ,
            VX_VELOX_CAPABILITY_BATCH_READ
        );
        assert_eq!(
            vx_velox_capabilities() & VX_VELOX_CAPABILITY_ARROW_SCHEMA,
            VX_VELOX_CAPABILITY_ARROW_SCHEMA
        );
        assert_eq!(
            vx_velox_capabilities() & VX_VELOX_CAPABILITY_ARRAY_ARROW_EXPORT,
            VX_VELOX_CAPABILITY_ARRAY_ARROW_EXPORT
        );
        assert_eq!(
            vx_velox_capabilities() & VX_VELOX_CAPABILITY_ROW_INDEX_PROJECTION,
            VX_VELOX_CAPABILITY_ROW_INDEX_PROJECTION
        );
        assert_eq!(
            vx_velox_capabilities() & VX_VELOX_CAPABILITY_NATURAL_SPLIT_PRUNING,
            VX_VELOX_CAPABILITY_NATURAL_SPLIT_PRUNING
        );
        assert_eq!(
            vx_velox_capabilities() & VX_VELOX_CAPABILITY_READ_CANCELLATION,
            VX_VELOX_CAPABILITY_READ_CANCELLATION
        );
        assert_eq!(
            vx_velox_capabilities() & VX_VELOX_CAPABILITY_EXPORT_CURSOR,
            VX_VELOX_CAPABILITY_EXPORT_CURSOR
        );
        assert_eq!(
            vx_velox_capabilities() & VX_VELOX_CAPABILITY_PLAIN_PROJECTION,
            VX_VELOX_CAPABILITY_PLAIN_PROJECTION
        );
        assert_eq!(
            vx_velox_capabilities() & VX_VELOX_CAPABILITY_VARBIN_VISITOR,
            VX_VELOX_CAPABILITY_VARBIN_VISITOR
        );
        assert_eq!(
            vx_velox_capabilities() & VX_VELOX_CAPABILITY_DICTIONARY_VISITOR,
            VX_VELOX_CAPABILITY_DICTIONARY_VISITOR
        );
        assert_eq!(
            vx_velox_capabilities() & VX_VELOX_CAPABILITY_CONSTANT_VISITOR,
            VX_VELOX_CAPABILITY_CONSTANT_VISITOR
        );
        assert_eq!(
            vx_velox_capabilities() & VX_VELOX_CAPABILITY_BOOL_VISITOR,
            VX_VELOX_CAPABILITY_BOOL_VISITOR
        );
        assert_eq!(
            vx_velox_capabilities() & VX_VELOX_CAPABILITY_DATE_VISITOR,
            VX_VELOX_CAPABILITY_DATE_VISITOR
        );
        assert_eq!(
            vx_velox_capabilities() & VX_VELOX_CAPABILITY_DECIMAL_VISITOR,
            VX_VELOX_CAPABILITY_DECIMAL_VISITOR
        );
        assert_eq!(
            vx_velox_capabilities() & VX_VELOX_CAPABILITY_STRUCT_VISITOR,
            VX_VELOX_CAPABILITY_STRUCT_VISITOR
        );
        assert_eq!(
            vx_velox_capabilities() & VX_VELOX_CAPABILITY_LIST_VISITOR,
            VX_VELOX_CAPABILITY_LIST_VISITOR
        );
        assert_eq!(
            vx_velox_capabilities() & VX_VELOX_CAPABILITY_MAP_VISITOR,
            VX_VELOX_CAPABILITY_MAP_VISITOR
        );
    }
}
