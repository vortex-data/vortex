// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::kernel::ParentKernelDense;
use vortex_array::kernel::ParentKernelEntry;
use vortex_array::kernel::ParentKernelSet;
use vortex_session::registry::CachedId;

use crate::BitPacked;

static KEYED_PARENT_KERNELS: [ParentKernelEntry<BitPacked>; 2] = [
    ParentKernelSet::lift_id(
        CachedId::new("vortex.filter"),
        &FilterExecuteAdaptor(BitPacked),
    ),
    ParentKernelSet::lift_id(CachedId::new("vortex.dict"), &TakeExecuteAdaptor(BitPacked)),
];

static KEYED_PARENT_KERNELS_DENSE: ParentKernelDense<BitPacked> = ParentKernelDense::new();

pub(crate) static PARENT_KERNELS: ParentKernelSet<BitPacked> =
    ParentKernelSet::new_indexed(&KEYED_PARENT_KERNELS, &KEYED_PARENT_KERNELS_DENSE, &[]);
