// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::registry::CachedId;

use crate::arrays::VarBin;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::filter::FilterExecuteAdaptor;
use crate::kernel::ParentKernelDense;
use crate::kernel::ParentKernelEntry;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::binary::CompareExecuteAdaptor;

static KEYED_PARENT_KERNELS: [ParentKernelEntry<VarBin>; 3] = [
    ParentKernelSet::lift_id(
        CachedId::new("vortex.binary"),
        &CompareExecuteAdaptor(VarBin),
    ),
    ParentKernelSet::lift_id(
        CachedId::new("vortex.filter"),
        &FilterExecuteAdaptor(VarBin),
    ),
    ParentKernelSet::lift_id(CachedId::new("vortex.dict"), &TakeExecuteAdaptor(VarBin)),
];

static KEYED_PARENT_KERNELS_DENSE: ParentKernelDense<VarBin> = ParentKernelDense::new();

pub(super) static PARENT_KERNELS: ParentKernelSet<VarBin> =
    ParentKernelSet::new_indexed(&KEYED_PARENT_KERNELS, &KEYED_PARENT_KERNELS_DENSE, &[]);
