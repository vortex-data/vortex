// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::registry::CachedId;

use crate::arrays::Patched;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::kernel::ParentKernelDense;
use crate::kernel::ParentKernelEntry;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::binary::CompareExecuteAdaptor;

static KEYED_PARENT_KERNELS: [ParentKernelEntry<Patched>; 2] = [
    ParentKernelSet::lift_id(
        CachedId::new("vortex.binary"),
        &CompareExecuteAdaptor(Patched),
    ),
    ParentKernelSet::lift_id(CachedId::new("vortex.dict"), &TakeExecuteAdaptor(Patched)),
];

static KEYED_PARENT_KERNELS_DENSE: ParentKernelDense<Patched> = ParentKernelDense::new();

pub(super) static PARENT_KERNELS: ParentKernelSet<Patched> =
    ParentKernelSet::new_indexed(&KEYED_PARENT_KERNELS, &KEYED_PARENT_KERNELS_DENSE, &[]);
