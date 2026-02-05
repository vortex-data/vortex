// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::BoolVTable;
use crate::arrays::filter::FilterExecuteAdaptor;
use crate::kernel::ParentKernelSet;

pub(super) const PARENT_KERNELS: ParentKernelSet<BoolVTable> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&FilterExecuteAdaptor(BoolVTable))]);
