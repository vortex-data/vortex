// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::kernel::ParentKernelSet;

use crate::RunEndVTable;

pub(super) const PARENT_KERNELS: ParentKernelSet<RunEndVTable> = ParentKernelSet::new(&[]);
