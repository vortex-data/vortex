// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::optimizer::rules::ParentRuleSet;

use crate::encodings::turboquant::TurboQuant;

pub(crate) static RULES: ParentRuleSet<TurboQuant> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&SliceReduceAdaptor(TurboQuant))]);

pub(crate) static PARENT_KERNELS: ParentKernelSet<TurboQuant> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&TakeExecuteAdaptor(TurboQuant))]);
