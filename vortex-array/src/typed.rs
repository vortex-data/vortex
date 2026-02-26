// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Typed and inner representations of arrays.
//!
//! - [`Array<V>`]: The public typed wrapper, parameterized by a concrete [`ScalarFnVTable`].
//! - [`ArrayInner<V>`]: The private inner struct that holds the vtable + options.
//! - [`DynArray`]: The private sealed trait for type-erased dispatch (bound, options in self).
