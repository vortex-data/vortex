// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::PrimitiveArray;
use crate::buffer::BufferHandle;
use crate::compute::conformance::filter::test_filter_conformance;
use crate::compute::conformance::mask::test_mask_conformance;
use crate::compute::conformance::search_sorted::rstest_reuse::apply;
use crate::compute::conformance::search_sorted::search_sorted_conformance;
use crate::compute::conformance::search_sorted::*;
use crate::dtype::PType;
use crate::search_sorted::SearchResult;
use crate::search_sorted::SearchSorted;
use crate::search_sorted::SearchSortedPrimitiveArray;
use crate::search_sorted::SearchSortedSide;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

#[apply(search_sorted_conformance)]
fn test_search_sorted_primitive(
    #[case] array: ArrayRef,
    #[case] value: i32,
    #[case] side: SearchSortedSide,
    #[case] expected: SearchResult,
) -> vortex_error::VortexResult<()> {
    let res = SearchSortedPrimitiveArray::<i32>::new(&array, &mut SESSION.create_execution_ctx())
        .search_sorted(&value, side)?;
    assert_eq!(res, expected);
    Ok(())
}

#[test]
fn test_mask_primitive_array() {
    test_mask_conformance(
        &PrimitiveArray::new(buffer![0, 1, 2, 3, 4], Validity::NonNullable).into_array(),
        &mut SESSION.create_execution_ctx(),
    );
    test_mask_conformance(
        &PrimitiveArray::new(buffer![0, 1, 2, 3, 4], Validity::AllValid).into_array(),
        &mut SESSION.create_execution_ctx(),
    );
    test_mask_conformance(
        &PrimitiveArray::new(buffer![0, 1, 2, 3, 4], Validity::AllInvalid).into_array(),
        &mut SESSION.create_execution_ctx(),
    );
    test_mask_conformance(
        &PrimitiveArray::new(
            buffer![0, 1, 2, 3, 4],
            Validity::Array(BoolArray::from_iter([true, false, true, false, true]).into_array()),
        )
        .into_array(),
        &mut SESSION.create_execution_ctx(),
    );
}

#[test]
fn test_filter_primitive_array() {
    // Test various sizes
    test_filter_conformance(
        &PrimitiveArray::new(buffer![42i32], Validity::NonNullable).into_array(),
        &mut SESSION.create_execution_ctx(),
    );
    test_filter_conformance(
        &PrimitiveArray::new(buffer![0, 1], Validity::NonNullable).into_array(),
        &mut SESSION.create_execution_ctx(),
    );
    test_filter_conformance(
        &PrimitiveArray::new(buffer![0, 1, 2, 3, 4], Validity::NonNullable).into_array(),
        &mut SESSION.create_execution_ctx(),
    );
    test_filter_conformance(
        &PrimitiveArray::new(buffer![0, 1, 2, 3, 4, 5, 6, 7], Validity::NonNullable).into_array(),
        &mut SESSION.create_execution_ctx(),
    );

    // Test with validity
    test_filter_conformance(
        &PrimitiveArray::new(buffer![0, 1, 2, 3, 4], Validity::AllValid).into_array(),
        &mut SESSION.create_execution_ctx(),
    );
    test_filter_conformance(
        &PrimitiveArray::new(
            buffer![0, 1, 2, 3, 4, 5],
            Validity::Array(
                BoolArray::from_iter([true, false, true, false, true, true]).into_array(),
            ),
        )
        .into_array(),
        &mut SESSION.create_execution_ctx(),
    );
}

#[test]
#[should_panic(expected = "PrimitiveArray buffer is not aligned")]
fn test_from_buffer_handle_rejects_misaligned_host_buffer() {
    let bytes = ByteBuffer::copy_from_aligned([0u8; 8], Alignment::of::<i32>());
    let misaligned = bytes.slice_unaligned(1..5);
    assert!(!misaligned.is_aligned(Alignment::of::<i32>()));

    PrimitiveArray::from_buffer_handle(
        BufferHandle::new_host(misaligned),
        PType::I32,
        Validity::NonNullable,
    );
}

#[test]
#[should_panic(expected = "not a multiple of element width")]
fn test_from_buffer_handle_rejects_length_not_multiple_of_ptype_width() {
    let bytes = ByteBuffer::copy_from_aligned([0u8; 5], Alignment::of::<i32>());

    PrimitiveArray::from_buffer_handle(
        BufferHandle::new_host(bytes),
        PType::I32,
        Validity::NonNullable,
    );
}
