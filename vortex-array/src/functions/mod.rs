// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod execution;
mod scalar;
mod session;
mod signature;
mod vtable;

pub use session::*;
pub use signature::*;
pub use vtable::*;

use arcref::ArcRef;

pub type FunctionId = ArcRef<str>;
