// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::registry::CachedId;

use crate::arrays::List;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::filter::FilterExecuteAdaptor;
use crate::kernel::ParentKernelDense;
use crate::kernel::ParentKernelEntry;
use crate::kernel::ParentKernelSet;

static KEYED_PARENT_KERNELS: [ParentKernelEntry<List>; 2] = [
    ParentKernelSet::lift_id(CachedId::new("vortex.filter"), &FilterExecuteAdaptor(List)),
    ParentKernelSet::lift_id(CachedId::new("vortex.dict"), &TakeExecuteAdaptor(List)),
];

static KEYED_PARENT_KERNELS_DENSE: ParentKernelDense<List> = ParentKernelDense::new();

pub(crate) static PARENT_KERNELS: ParentKernelSet<List> =
    ParentKernelSet::new_indexed(&KEYED_PARENT_KERNELS, &KEYED_PARENT_KERNELS_DENSE, &[]);
