// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::registry::CachedId;

use crate::arrays::FixedSizeList;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::kernel::ParentKernelDense;
use crate::kernel::ParentKernelEntry;
use crate::kernel::ParentKernelSet;

static KEYED_PARENT_KERNELS: [ParentKernelEntry<FixedSizeList>; 1] = [ParentKernelSet::lift_id(
    CachedId::new("vortex.dict"),
    &TakeExecuteAdaptor(FixedSizeList),
)];

static KEYED_PARENT_KERNELS_DENSE: ParentKernelDense<FixedSizeList> = ParentKernelDense::new();

pub(crate) static PARENT_KERNELS: ParentKernelSet<FixedSizeList> =
    ParentKernelSet::new_indexed(&KEYED_PARENT_KERNELS, &KEYED_PARENT_KERNELS_DENSE, &[]);
