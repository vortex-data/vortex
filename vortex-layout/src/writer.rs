// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::future::BoxFuture;
use vortex_error::VortexResult;

use crate::LayoutRef;

/// A future created by a strategy to yield a layout. It is its own
/// trait to be potentially extended with new methods.
// Tag for Python docs:
// [layout writer]
pub type SendableLayoutFuture = BoxFuture<'static, VortexResult<LayoutRef>>;
// [layout writer]
