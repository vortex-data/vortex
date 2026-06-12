// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Common test utilities for ListView tests.

use std::sync::LazyLock;

use vortex_buffer::buffer;
use vortex_session::VortexSession;

use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::session::ArraySession;
use crate::validity::Validity;

/// A shared session for `ListView` tests, used to create execution contexts via
/// [`create_execution_ctx`](crate::VortexSessionExecute::create_execution_ctx).
pub static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

/// Creates a basic ListView for testing: [[0,1,2], [3,4], [5,6], [7,8,9]]
pub fn create_basic_listview() -> ListViewArray {
    let elements = buffer![0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_array();
    let offsets = buffer![0u32, 3, 5, 7].into_array();
    let sizes = buffer![3u32, 2, 2, 3].into_array();
    unsafe {
        ListViewArray::new_unchecked(elements, offsets, sizes, Validity::NonNullable)
            .with_zero_copy_to_list(true)
    }
}

/// Creates a sparse ListView with two overlap regions
/// `[[0,1,2], [1,2], [18, 19], [19]]` over 20 elements.
pub fn create_sparse_overlapping_listview() -> ListViewArray {
    let elements = buffer![0i32..20].into_array();
    let offsets = buffer![0u32, 1, 18, 19].into_array();
    let sizes = buffer![3u32, 2, 2, 1].into_array();
    ListViewArray::new(elements, offsets, sizes, Validity::NonNullable)
}

/// Creates a nullable ListView: [[10,20], null, [50]]
pub fn create_nullable_listview() -> ListViewArray {
    let elements = buffer![10i32, 20, 30, 40, 50].into_array();
    let offsets = buffer![0u32, 2, 4].into_array();
    let sizes = buffer![2u32, 2, 1].into_array();
    let validity = Validity::Array(BoolArray::from_iter(vec![true, false, true]).into_array());
    unsafe {
        ListViewArray::new_unchecked(elements, offsets, sizes, validity)
            .with_zero_copy_to_list(true)
    }
}

/// Creates a ListView with empty lists: [[], [], [], []]
pub fn create_empty_lists_listview() -> ListViewArray {
    let elements = buffer![99i32].into_array();
    let offsets = buffer![0u32, 0, 0, 0].into_array();
    let sizes = buffer![0u32, 0, 0, 0].into_array();
    unsafe {
        ListViewArray::new_unchecked(elements, offsets, sizes, Validity::NonNullable)
            .with_zero_copy_to_list(true)
    }
}

/// Creates a ListView with empty lists and elements: [[]]
pub fn create_empty_elements_listview() -> ListViewArray {
    let elements = PrimitiveArray::from_iter::<[i32; 0]>([]).into_array();
    let offsets = buffer![0u32; 0].into_array();
    let sizes = buffer![0u32; 0].into_array();
    unsafe {
        ListViewArray::new_unchecked(elements, offsets, sizes, Validity::NonNullable)
            .with_zero_copy_to_list(true)
    }
}

/// Creates a ListView with overlapping lists and out-of-order offsets
/// Lists: [[5,6,7], [2,3], [8,9], [0,1], [1,2,3,4]]
pub fn create_overlapping_listview() -> ListViewArray {
    let elements = buffer![0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_array();
    let offsets = buffer![5u32, 2, 8, 0, 1].into_array();
    let sizes = buffer![3u32, 2, 2, 2, 4].into_array();
    ListViewArray::new(elements, offsets, sizes, Validity::NonNullable)
}

/// Creates a large ListView for performance testing
pub fn create_large_listview() -> ListViewArray {
    let elements = PrimitiveArray::from_iter(0i32..1000).into_array();
    let offsets = buffer![0u32, 100, 200, 300, 400, 500, 600, 700, 800, 900].into_array();
    let sizes = buffer![50u32, 50, 50, 50, 50, 50, 50, 50, 50, 50].into_array();
    ListViewArray::new(elements, offsets, sizes, Validity::NonNullable)
}
