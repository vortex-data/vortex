// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::VortexSessionDefault;
use vortex::io::session::RuntimeSessionBuilderExt;
use vortex::session::VortexSession;
use vortex_tui::launch;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let session = VortexSession::default_builder()
        .with_tokio()
        .allow_unknown()
        .build();
    if let Err(err) = launch(&session).await {
        // Defer help/version/usage errors back to clap so their formatting
        // and exit codes match the standalone-binary convention exactly.
        if let Some(clap_err) = err.downcast_ref::<clap::Error>() {
            clap_err.exit();
        }
        return Err(err);
    }
    Ok(())
}
