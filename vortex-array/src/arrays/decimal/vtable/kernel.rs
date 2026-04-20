// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::registry::CachedId;

use crate::arrays::Decimal;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::kernel::ParentKernelDense;
use crate::kernel::ParentKernelEntry;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::between::BetweenExecuteAdaptor;
use crate::scalar_fn::fns::cast::CastExecuteAdaptor;
use crate::scalar_fn::fns::fill_null::FillNullExecuteAdaptor;

static KEYED_PARENT_KERNELS: [ParentKernelEntry<Decimal>; 4] = [
    ParentKernelSet::lift_id(
        CachedId::new("vortex.between"),
        &BetweenExecuteAdaptor(Decimal),
    ),
    ParentKernelSet::lift_id(CachedId::new("vortex.cast"), &CastExecuteAdaptor(Decimal)),
    ParentKernelSet::lift_id(
        CachedId::new("vortex.fill_null"),
        &FillNullExecuteAdaptor(Decimal),
    ),
    ParentKernelSet::lift_id(CachedId::new("vortex.dict"), &TakeExecuteAdaptor(Decimal)),
];

static KEYED_PARENT_KERNELS_DENSE: ParentKernelDense<Decimal> = ParentKernelDense::new();

pub(super) static PARENT_KERNELS: ParentKernelSet<Decimal> =
    ParentKernelSet::new_indexed(&KEYED_PARENT_KERNELS, &KEYED_PARENT_KERNELS_DENSE, &[]);
