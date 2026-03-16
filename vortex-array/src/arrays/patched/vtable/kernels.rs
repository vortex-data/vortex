// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::PatchedVTable;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::binary::CompareExecuteAdaptor;

pub(super) const PARENT_KERNELS: ParentKernelSet<PatchedVTable> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&CompareExecuteAdaptor(PatchedVTable))]);
