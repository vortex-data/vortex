// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::kernel::ParentKernelDense;
use vortex_array::kernel::ParentKernelEntry;
use vortex_array::kernel::ParentKernelSet;
use vortex_session::registry::CachedId;

use crate::ZigZag;

static KEYED_PARENT_KERNELS: [ParentKernelEntry<ZigZag>; 1] = [ParentKernelSet::lift_id(
    CachedId::new("vortex.dict"),
    &TakeExecuteAdaptor(ZigZag),
)];

static KEYED_PARENT_KERNELS_DENSE: ParentKernelDense<ZigZag> = ParentKernelDense::new();

pub(crate) static PARENT_KERNELS: ParentKernelSet<ZigZag> =
    ParentKernelSet::new_indexed(&KEYED_PARENT_KERNELS, &KEYED_PARENT_KERNELS_DENSE, &[]);
