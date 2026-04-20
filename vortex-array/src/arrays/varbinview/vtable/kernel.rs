// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::registry::CachedId;

use crate::arrays::VarBinView;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::kernel::ParentKernelDense;
use crate::kernel::ParentKernelEntry;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::zip::ZipExecuteAdaptor;

static KEYED_PARENT_KERNELS: [ParentKernelEntry<VarBinView>; 2] = [
    ParentKernelSet::lift_id(
        CachedId::new("vortex.dict"),
        &TakeExecuteAdaptor(VarBinView),
    ),
    ParentKernelSet::lift_id(CachedId::new("vortex.zip"), &ZipExecuteAdaptor(VarBinView)),
];

static KEYED_PARENT_KERNELS_DENSE: ParentKernelDense<VarBinView> = ParentKernelDense::new();

pub(super) static PARENT_KERNELS: ParentKernelSet<VarBinView> =
    ParentKernelSet::new_indexed(&KEYED_PARENT_KERNELS, &KEYED_PARENT_KERNELS_DENSE, &[]);
