// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::arrays::{ListViewArray, ListViewVTable};

// TODO(connor)[ListView]: Once `ListViewArray` replaces `ListArray` as the default `List` encoding,
// we can remove this and simply use `to_list` via `ToCanonical`.
/// Helper trait to extract ListViewArray from ArrayRef.
trait ToListView {
    fn to_listview(&self) -> ListViewArray;
}

impl ToListView for ArrayRef {
    fn to_listview(&self) -> ListViewArray {
        self.as_opt::<ListViewVTable>()
            .unwrap_or_else(|| vortex_panic!("Expected ListViewArray"))
            .clone()
    }
}

mod basic;
mod common;
mod filter;
mod nested;
mod nullability;
mod operations;
mod take;
