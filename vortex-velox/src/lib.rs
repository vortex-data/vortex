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
mod visitor;

pub use api::*;
pub use array::vx_velox_array_export_arrow;
pub use array::vx_velox_array_get_field;
pub use array::vx_velox_array_invalid_count;
pub use array::vx_velox_arrow_memory_callbacks;
pub use projection::vx_velox_expression_select_with_row_index;
pub use read_at::vx_velox_buffer;
pub use read_at::vx_velox_read_at;
pub use read_at::vx_velox_read_at_callbacks;
pub use read_at::vx_velox_read_request;
pub use schema::vx_velox_source_export_schema;
pub use source::vx_velox_natural_split;
pub use source::vx_velox_source;
pub use source::vx_velox_source_prune_natural_splits;
pub use visitor::vx_velox_buffer_owner;
pub use visitor::vx_velox_primitive_type;
pub use visitor::vx_velox_primitive_view;
pub use visitor::vx_velox_validity_kind;
pub use visitor::vx_velox_visit_request;
pub use visitor::vx_velox_visitor;

/// The current major version of the Vortex and Velox adapter ABI.
pub const VX_VELOX_ABI_VERSION: u32 = 1;

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
    }
}
