// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::scalar_fn::fns::binary::CompareExecuteAdaptor;
use vortex_array::scalar_fn::fns::cast::CastExecuteAdaptor;

use crate::OnPair;

// Compare is pushed: LPM-tokenise the literal once, compare the row's
// `codes[lo..hi]` against the token sequence as `&[u16]` — no decode.
// Like is currently *not* registered: the per-row byte-streaming /
// `memmem`-on-decoded-row implementations are slower than letting the
// canonicalize + scalar `LIKE` path run. A token-DFA pushdown (FSST-
// style) is the right replacement and tracked as future work.
pub(super) const PARENT_KERNELS: ParentKernelSet<OnPair> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CastExecuteAdaptor(OnPair)),
    ParentKernelSet::lift(&CompareExecuteAdaptor(OnPair)),
    ParentKernelSet::lift(&FilterExecuteAdaptor(OnPair)),
]);
