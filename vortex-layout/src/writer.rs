// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::future::BoxFuture;
use vortex_error::VortexResult;

use crate::LayoutRef;

/// A Future that resolves to a layout. Returned from a `LayoutStrategy` after it finishes
/// assembling the segments for the writer.
// Tag for Python docs:
// [layout writer]
pub type SendableLayoutFuture = BoxFuture<'static, VortexResult<LayoutRef>>;
// [layout writer]
