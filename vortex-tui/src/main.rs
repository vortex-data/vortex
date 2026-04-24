// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::VortexSessionDefault;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex_tui::launch;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let session = VortexSession::default().with_tokio().allow_unknown();
    launch(&session).await
}
