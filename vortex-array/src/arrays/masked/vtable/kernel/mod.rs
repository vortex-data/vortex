// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod filter;

use crate::arrays::MaskedVTable;
use crate::arrays::masked::vtable::kernel::filter::MaskedFilterKernel;
use crate::kernel::ParentKernelSet;

pub(super) const PARENT_KERNELS: ParentKernelSet<MaskedVTable> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&MaskedFilterKernel)]);
