// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::registry::CachedId;

use crate::arrays::Struct;
use crate::kernel::ParentKernelDense;
use crate::kernel::ParentKernelEntry;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::cast::CastExecuteAdaptor;
use crate::scalar_fn::fns::zip::ZipExecuteAdaptor;

static KEYED_PARENT_KERNELS: [ParentKernelEntry<Struct>; 2] = [
    ParentKernelSet::lift_id(CachedId::new("vortex.cast"), &CastExecuteAdaptor(Struct)),
    ParentKernelSet::lift_id(CachedId::new("vortex.zip"), &ZipExecuteAdaptor(Struct)),
];

static KEYED_PARENT_KERNELS_DENSE: ParentKernelDense<Struct> = ParentKernelDense::new();

pub(super) static PARENT_KERNELS: ParentKernelSet<Struct> =
    ParentKernelSet::new_indexed(&KEYED_PARENT_KERNELS, &KEYED_PARENT_KERNELS_DENSE, &[]);
