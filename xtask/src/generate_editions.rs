// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;

use anyhow::Context;

/// Regenerate the edition artifacts under `docs/specs/editions/` (per-edition JSON files and
/// Markdown pages) and the generated index block in `docs/specs/editions.md`, from the
/// definitions in the `vortex-edition` crate.
pub(crate) fn generate_editions() -> anyhow::Result<()> {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .context("xtask manifest dir has no parent")?;
    let written =
        vortex_edition::generate::generate(repo_root).context("generating edition artifacts")?;
    for path in written {
        println!("wrote {path}");
    }
    Ok(())
}
