// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

pub trait RowSelection {
    fn is_disjoint(&self, range: &Range<u64>) -> bool;

    fn slice(&self, range: &Range<u64>) -> Arc<dyn RowSelection>;
}

pub type RowSelectionRef = Arc<dyn RowSelection>;
