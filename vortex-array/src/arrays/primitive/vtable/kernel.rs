// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::registry::CachedId;

use crate::arrays::Primitive;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::kernel::ParentKernelDense;
use crate::kernel::ParentKernelEntry;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::between::BetweenExecuteAdaptor;
use crate::scalar_fn::fns::cast::CastExecuteAdaptor;
use crate::scalar_fn::fns::fill_null::FillNullExecuteAdaptor;

static KEYED_PARENT_KERNELS: [ParentKernelEntry<Primitive>; 4] = [
    ParentKernelSet::lift_id(
        CachedId::new("vortex.between"),
        &BetweenExecuteAdaptor(Primitive),
    ),
    ParentKernelSet::lift_id(CachedId::new("vortex.cast"), &CastExecuteAdaptor(Primitive)),
    ParentKernelSet::lift_id(
        CachedId::new("vortex.fill_null"),
        &FillNullExecuteAdaptor(Primitive),
    ),
    ParentKernelSet::lift_id(CachedId::new("vortex.dict"), &TakeExecuteAdaptor(Primitive)),
];

static KEYED_PARENT_KERNELS_DENSE: ParentKernelDense<Primitive> = ParentKernelDense::new();

pub(super) static PARENT_KERNELS: ParentKernelSet<Primitive> =
    ParentKernelSet::new_indexed(&KEYED_PARENT_KERNELS, &KEYED_PARENT_KERNELS_DENSE, &[]);
