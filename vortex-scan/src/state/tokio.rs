// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A scan driver designed for engines that use a Tokio runtime for orchestrating work, including
//! CPU-bound tasks. Let's not beat around the bush, this is targeted at DataFusion :)

use crate::state::Scan2;
use tokio::runtime::Handle;

impl Scan2 {
    pub fn into_tokio_steam(self, _handle: Handle) {}
}
