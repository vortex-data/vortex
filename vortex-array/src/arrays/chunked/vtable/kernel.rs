// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::ChunkedVTable;
use crate::arrays::filter::FilterExecuteAdaptor;
use crate::kernel::ParentKernelSet;

pub(super) const PARENT_KERNELS: ParentKernelSet<ChunkedVTable> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&FilterExecuteAdaptor(ChunkedVTable))]);
