// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod execution;
pub mod scalar;
mod session;
mod signature;
pub mod v2;
mod vtable;

use arcref::ArcRef;
pub use execution::*;
pub use session::*;
pub use signature::*;
pub use vtable::*;

pub type FunctionId = ArcRef<str>;
