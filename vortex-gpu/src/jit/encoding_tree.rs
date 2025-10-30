// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

pub type EncodingTreeRef = Arc<dyn EncodingTree + 'static>;

pub trait EncodingTree {
    fn as_any(&self) -> &dyn Any;
}
