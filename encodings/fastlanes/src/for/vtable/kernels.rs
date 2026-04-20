// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::kernel::ParentKernelDense;
use vortex_array::kernel::ParentKernelEntry;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::scalar_fn::fns::binary::CompareExecuteAdaptor;
use vortex_session::registry::CachedId;

use crate::FoR;

static KEYED_PARENT_KERNELS: [ParentKernelEntry<FoR>; 2] = [
    ParentKernelSet::lift_id(CachedId::new("vortex.binary"), &CompareExecuteAdaptor(FoR)),
    ParentKernelSet::lift_id(CachedId::new("vortex.dict"), &TakeExecuteAdaptor(FoR)),
];

static KEYED_PARENT_KERNELS_DENSE: ParentKernelDense<FoR> = ParentKernelDense::new();

pub(crate) static PARENT_KERNELS: ParentKernelSet<FoR> =
    ParentKernelSet::new_indexed(&KEYED_PARENT_KERNELS, &KEYED_PARENT_KERNELS_DENSE, &[]);
