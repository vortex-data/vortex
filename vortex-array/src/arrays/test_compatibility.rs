//! Metadata serialization tests for all builtin array types.
//!
//! This test suite checks for when metadata format changes for one of the core
//! encodings.
//!
//! These tests can be run as normal using `cargo test`. Failures will occur when the serialization
//! format of any encoding metadata changes.
//!
//! You can update the stored "golden files" by re-running your test command with the
//! `UPDATE_GOLDENFILES=1` environment variable. For example:
//!
//! ```ignored
//! $ UPDATE_GOLDENFILES=1 cargo test -p vortex-array --tests test_compatibility
//! ````

use vortex_dtype::{Nullability, PType};
use vortex_scalar::Scalar;

use crate::arrays::{
    BoolMetadata, ChunkedMetadata, ConstantMetadata, ListMetadata, NullMetadata, PrimitiveMetadata,
    SparseMetadata, StructMetadata, VarBinMetadata, VarBinViewMetadata,
};
use crate::patches::PatchesMetadata;
use crate::test_harness::check_metadata;
use crate::validity::ValidityMetadata;

#[cfg_attr(miri, ignore)]
#[test]
fn test_bool_metadata() {
    check_metadata(
        "bool.metadata",
        SerdeMetadata(BoolMetadata {
            validity: ValidityMetadata::AllValid,
            first_byte_bit_offset: u8::MAX,
        }),
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_chunked_metadata() {
    check_metadata("chunked.metadata", ChunkedMetadata { nchunks: 1 });
}

// #[cfg_attr(miri, ignore)]
// #[test]
// fn test_constant_metadata() {
//     check_metadata(
//         "constant.metadata",
//         ConstantMetadata {
//             scalar_value: Scalar::primitive(i32::MAX, Nullability::Nullable).into_value(),
//         },
//     );
// }

#[cfg_attr(miri, ignore)]
#[test]
fn test_list_metadata() {
    check_metadata(
        "list.metadata",
        SerdeMetadata(ListMetadata {
            validity: ValidityMetadata::AllValid,
            elements_len: usize::MAX,
            offset_ptype: PType::U64,
        }),
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_null_metadata() {
    check_metadata("null.metadata", NullMetadata);
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_primitive_metadata() {
    check_metadata(
        "primitive.metadata",
        SerdeMetadata(PrimitiveMetadata {
            validity: ValidityMetadata::NonNullable,
        }),
    );
}

// #[cfg_attr(miri, ignore)]
// #[test]
// fn test_sparse_metadata() {
//     check_metadata(
//         "sparse.metadata",
//         SparseMetadata {
//             fill_value: Scalar::primitive(i32::MAX, Nullability::NonNullable).into_value(),
//             patches: PatchesMetadata::new(usize::MAX, PType::U64),
//             indices_offset: usize::MAX,
//         },
//     );
// }

#[cfg_attr(miri, ignore)]
#[test]
fn test_struct_metadata() {
    check_metadata(
        "struct.metadata",
        SerdeMetadata(StructMetadata {
            validity: ValidityMetadata::AllValid,
        }),
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_varbin_metadata() {
    check_metadata(
        "varbin.metadata",
        SerdeMetadata(VarBinMetadata {
            validity: ValidityMetadata::AllValid,
            bytes_len: usize::MAX,
            offsets_ptype: PType::U64,
        }),
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_varbinview_metadata() {
    check_metadata(
        "varbinview.metadata",
        VarBinViewMetadata {
            buffer_lens: vec![1, 2, 3, 4],
            validity: ValidityMetadata::AllValid,
        },
    );
}
