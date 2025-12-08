// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_mask::Mask;

use crate::ArrayRef;
use crate::stats::ArrayStats;

#[derive(Clone, Debug)]
pub struct FilterArray {
    pub(super) child: ArrayRef,
    pub(super) mask: Mask,
    pub(super) stats: ArrayStats,
}
