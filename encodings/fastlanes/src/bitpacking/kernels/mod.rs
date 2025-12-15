// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::kernel::ParentKernelSet;

use crate::BitPackedVTable;

pub(super) const PARENT_KERNELS: ParentKernelSet<BitPackedVTable> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&filter::BitPackingFilterKernel)]);

mod filter;
