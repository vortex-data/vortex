// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::arrays::slice::SliceExecuteAdaptor;
use vortex_array::kernel::ParentKernelDense;
use vortex_array::kernel::ParentKernelEntry;
use vortex_array::kernel::ParentKernelSet;
use vortex_session::registry::CachedId;

use crate::alp_rd::ALPRD;

static KEYED_PARENT_KERNELS: [ParentKernelEntry<ALPRD>; 3] = [
    ParentKernelSet::lift_id(CachedId::new("vortex.slice"), &SliceExecuteAdaptor(ALPRD)),
    ParentKernelSet::lift_id(CachedId::new("vortex.filter"), &FilterExecuteAdaptor(ALPRD)),
    ParentKernelSet::lift_id(CachedId::new("vortex.dict"), &TakeExecuteAdaptor(ALPRD)),
];

static KEYED_PARENT_KERNELS_DENSE: ParentKernelDense<ALPRD> = ParentKernelDense::new();

pub(crate) static PARENT_KERNELS: ParentKernelSet<ALPRD> =
    ParentKernelSet::new_indexed(&KEYED_PARENT_KERNELS, &KEYED_PARENT_KERNELS_DENSE, &[]);
