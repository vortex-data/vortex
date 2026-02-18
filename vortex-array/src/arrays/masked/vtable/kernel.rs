// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::MaskedVTable;
use crate::arrays::TakeExecuteAdaptor;
use crate::kernel::ParentKernelSet;

// TODO(joe): add CompareExecuteAdaptor to push comparisons through the mask without canonicalizing.
pub(super) const PARENT_KERNELS: ParentKernelSet<MaskedVTable> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&TakeExecuteAdaptor(MaskedVTable))]);
