// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::arrays::dict::TakeExecuteAdaptor;
use vortex::array::arrays::slice::SliceReduceAdaptor;
use vortex::array::kernel::ParentKernelSet;
use vortex::array::optimizer::rules::ParentRuleSet;

use crate::encodings::turboquant::array::TurboQuant;

pub(crate) static RULES: ParentRuleSet<TurboQuant> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&SliceReduceAdaptor(TurboQuant))]);

pub(crate) static PARENT_KERNELS: ParentKernelSet<TurboQuant> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&TakeExecuteAdaptor(TurboQuant))]);
