// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::Chunked;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::filter::FilterExecuteAdaptor;
use crate::arrays::slice::SliceExecuteAdaptor;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::mask::MaskExecuteAdaptor;
use crate::scalar_fn::fns::zip::ZipExecuteAdaptor;

pub(crate) static PARENT_KERNELS: ParentKernelSet<Chunked> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&FilterExecuteAdaptor(Chunked)),
    ParentKernelSet::lift(&MaskExecuteAdaptor(Chunked)),
    ParentKernelSet::lift(&SliceExecuteAdaptor(Chunked)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(Chunked)),
    ParentKernelSet::lift(&ZipExecuteAdaptor(Chunked)),
]);
