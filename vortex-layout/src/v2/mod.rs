// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An updated Vortex Layouts API to power the new lazy array evaluation engine.
//!
//! This is currently highly experimental and subject to change. For anyone who has written custom
//! layouts, we will provide migration guidance when this API stabilizes.

mod layout;
mod layouts;
mod optimizer;
mod session;
mod view;
mod vtable;
