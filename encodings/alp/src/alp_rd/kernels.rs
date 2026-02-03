// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::FilterReduceRule;
use vortex_array::kernel::ParentKernelSet;

use crate::ALPRDVTable;

pub(super) const PARENT_KERNELS: ParentKernelSet<ALPRDVTable> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&FilterReduceRule)]);
