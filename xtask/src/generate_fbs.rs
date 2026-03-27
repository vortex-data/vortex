// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use anyhow::Context;
use xshell::Shell;
use xshell::cmd;

static PLANUS_BIN: &str = "planus";
static SCHEMAS: &[(&str, &str)] = &[
    (
        "./flatbuffers/vortex-array/array.fbs",
        "./src/generated/array.rs",
    ),
    (
        "./flatbuffers/vortex-dtype/dtype.fbs",
        "./src/generated/dtype.rs",
    ),
    (
        "./flatbuffers/vortex-file/footer.fbs",
        "./src/generated/footer.rs",
    ),
    (
        "./flatbuffers/vortex-layout/layout.fbs",
        "./src/generated/layout.rs",
    ),
    (
        "./flatbuffers/vortex-serde/message.fbs",
        "./src/generated/message.rs",
    ),
];

pub fn generate_fbs() -> anyhow::Result<()> {
    let sh = Shell::new()?;

    // CD to vortex-flatbuffers project
    sh.change_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../vortex-flatbuffers"));

    for (schema, output) in SCHEMAS {
        cmd!(sh, "{PLANUS_BIN} rust --format false -o {output} {schema}")
            .run()
            .with_context(|| {
                format!(
                    "failed to run `{PLANUS_BIN}` for {schema}; install it with `cargo install planus-cli`"
                )
            })?;
    }

    Ok(())
}
