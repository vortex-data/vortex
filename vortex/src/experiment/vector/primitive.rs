// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::experiment::N;
use crate::experiment::bits::BitVector;
use std::sync::Arc;

pub struct PrimitiveVector<T> {
    elements: Arc<[T; N]>,
    validity: Option<BitVector>, // TODO(ngates): is this optional? Or just always allocate?
}
