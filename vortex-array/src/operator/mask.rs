// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::pin::Pin;
use std::task::{Context, Poll};
use vortex_mask::Mask;

pub struct MaskExecution {}

impl Future for MaskExecution {
    type Output = Mask;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        todo!()
    }
}
