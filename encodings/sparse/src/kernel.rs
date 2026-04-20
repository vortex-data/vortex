// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::arrays::slice::SliceExecuteAdaptor;
use vortex_array::kernel::ParentKernelDense;
use vortex_array::kernel::ParentKernelEntry;
use vortex_array::kernel::ParentKernelSet;
use vortex_session::registry::CachedId;

use crate::Sparse;

static KEYED_PARENT_KERNELS: [ParentKernelEntry<Sparse>; 3] = [
    ParentKernelSet::lift_id(
        CachedId::new("vortex.filter"),
        &FilterExecuteAdaptor(Sparse),
    ),
    ParentKernelSet::lift_id(CachedId::new("vortex.slice"), &SliceExecuteAdaptor(Sparse)),
    ParentKernelSet::lift_id(CachedId::new("vortex.dict"), &TakeExecuteAdaptor(Sparse)),
];

static KEYED_PARENT_KERNELS_DENSE: ParentKernelDense<Sparse> = ParentKernelDense::new();

pub(crate) static PARENT_KERNELS: ParentKernelSet<Sparse> =
    ParentKernelSet::new_indexed(&KEYED_PARENT_KERNELS, &KEYED_PARENT_KERNELS_DENSE, &[]);
