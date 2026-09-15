// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row access over arrays.
//!
//! [`ArrayProbe`] is a borrowed reader: a one-off read that retains nothing, or a borrow of a
//! [`RepeatedArrayProbe`], which owns its array and keeps encoding state, validity and child
//! probes between reads. Encodings implement a single
//! [`probe_scalar`](crate::vtable::OperationsVTable::probe_scalar) that serves both through
//! [`ProbeState`].

mod array;
pub use array::*;
mod repeated;
pub use repeated::*;
