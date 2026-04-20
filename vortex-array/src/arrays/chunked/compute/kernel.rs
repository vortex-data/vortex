// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::registry::CachedId;

use crate::arrays::Chunked;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::filter::FilterExecuteAdaptor;
use crate::arrays::slice::SliceExecuteAdaptor;
use crate::kernel::ParentKernelDense;
use crate::kernel::ParentKernelEntry;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::mask::MaskExecuteAdaptor;
use crate::scalar_fn::fns::zip::ZipExecuteAdaptor;

static KEYED_PARENT_KERNELS: [ParentKernelEntry<Chunked>; 5] = [
    ParentKernelSet::lift_id(
        CachedId::new("vortex.filter"),
        &FilterExecuteAdaptor(Chunked),
    ),
    ParentKernelSet::lift_id(CachedId::new("vortex.mask"), &MaskExecuteAdaptor(Chunked)),
    ParentKernelSet::lift_id(CachedId::new("vortex.slice"), &SliceExecuteAdaptor(Chunked)),
    ParentKernelSet::lift_id(CachedId::new("vortex.dict"), &TakeExecuteAdaptor(Chunked)),
    ParentKernelSet::lift_id(CachedId::new("vortex.zip"), &ZipExecuteAdaptor(Chunked)),
];

static KEYED_PARENT_KERNELS_DENSE: ParentKernelDense<Chunked> = ParentKernelDense::new();

pub(crate) static PARENT_KERNELS: ParentKernelSet<Chunked> =
    ParentKernelSet::new_indexed(&KEYED_PARENT_KERNELS, &KEYED_PARENT_KERNELS_DENSE, &[]);
