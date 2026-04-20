// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::registry::CachedId;

use crate::arrays::Bool;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::kernel::ParentKernelDense;
use crate::kernel::ParentKernelEntry;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::fill_null::FillNullExecuteAdaptor;

static KEYED_PARENT_KERNELS: [ParentKernelEntry<Bool>; 2] = [
    ParentKernelSet::lift_id(
        CachedId::new("vortex.fill_null"),
        &FillNullExecuteAdaptor(Bool),
    ),
    ParentKernelSet::lift_id(CachedId::new("vortex.dict"), &TakeExecuteAdaptor(Bool)),
];

static KEYED_PARENT_KERNELS_DENSE: ParentKernelDense<Bool> = ParentKernelDense::new();

pub(super) static PARENT_KERNELS: ParentKernelSet<Bool> =
    ParentKernelSet::new_indexed(&KEYED_PARENT_KERNELS, &KEYED_PARENT_KERNELS_DENSE, &[]);
