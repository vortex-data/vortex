// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::registry::CachedId;

use crate::arrays::Dict;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::kernel::ParentKernelDense;
use crate::kernel::ParentKernelEntry;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::binary::CompareExecuteAdaptor;
use crate::scalar_fn::fns::fill_null::FillNullExecuteAdaptor;

static KEYED_PARENT_KERNELS: [ParentKernelEntry<Dict>; 3] = [
    ParentKernelSet::lift_id(CachedId::new("vortex.binary"), &CompareExecuteAdaptor(Dict)),
    ParentKernelSet::lift_id(CachedId::new("vortex.dict"), &TakeExecuteAdaptor(Dict)),
    ParentKernelSet::lift_id(
        CachedId::new("vortex.fill_null"),
        &FillNullExecuteAdaptor(Dict),
    ),
];

static KEYED_PARENT_KERNELS_DENSE: ParentKernelDense<Dict> = ParentKernelDense::new();

pub(super) static PARENT_KERNELS: ParentKernelSet<Dict> =
    ParentKernelSet::new_indexed(&KEYED_PARENT_KERNELS, &KEYED_PARENT_KERNELS_DENSE, &[]);
