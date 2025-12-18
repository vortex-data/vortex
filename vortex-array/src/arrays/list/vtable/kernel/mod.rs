// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod filter;

use crate::arrays::ListVTable;
use crate::arrays::list::vtable::kernel::filter::ListFilterKernel;
use crate::kernel::ParentKernelSet;

pub(super) const PARENT_KERNELS: ParentKernelSet<ListVTable> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&ListFilterKernel)]);
