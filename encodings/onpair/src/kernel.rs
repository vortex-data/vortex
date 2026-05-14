// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::scalar_fn::fns::binary::CompareExecuteAdaptor;
use vortex_array::scalar_fn::fns::cast::CastExecuteAdaptor;
use vortex_array::scalar_fn::fns::like::LikeExecuteAdaptor;

use crate::OnPair;

// Compare:  LPM-tokenise the literal once, compare row codes as &[u16].
// Like:     OnPair-style PrefixAutomaton for `prefix%`, dict-bloom +
//           memmem for `%substring%`, and token-equality for `'literal'`.
//           See encodings/onpair/src/dfa.rs and compute/like.rs.
pub(super) const PARENT_KERNELS: ParentKernelSet<OnPair> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CastExecuteAdaptor(OnPair)),
    ParentKernelSet::lift(&CompareExecuteAdaptor(OnPair)),
    ParentKernelSet::lift(&FilterExecuteAdaptor(OnPair)),
    ParentKernelSet::lift(&LikeExecuteAdaptor(OnPair)),
]);
