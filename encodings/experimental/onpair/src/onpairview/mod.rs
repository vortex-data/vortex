// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`ListView`](vortex_array::arrays::ListViewArray)-shaped sibling of the
//! [`OnPair`](crate::OnPair) encoding. See [`OnPairView`] for the full rationale.

mod array;
mod canonical;
mod compute;
mod kernel;
mod ops;
mod rules;
#[cfg(test)]
mod tests;

pub use array::*;
pub use canonical::OnPairViewDecodeMode;
pub use canonical::SPAN_DECODE_DENSITY_THRESHOLD;
pub use canonical::canonicalize_to_varbin;
pub use canonical::canonicalize_with;
pub use canonical::compact;
