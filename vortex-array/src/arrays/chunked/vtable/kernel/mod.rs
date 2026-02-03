// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod filter;

use crate::arrays::ChunkedVTable;
use crate::arrays::chunked::vtable::kernel::filter::ChunkedFilterKernel;
use crate::kernel::ParentKernelSet;

pub(super) const PARENT_KERNELS: ParentKernelSet<ChunkedVTable> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&ChunkedFilterKernel)]);
