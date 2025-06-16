use std::path::PathBuf;

use xshell::{Shell, cmd};

static FLATC_BIN: &str = "flatc";

pub fn generate_fbs() -> anyhow::Result<()> {
    let sh = Shell::new()?;

    let files = vec![
        "./flatbuffers/vortex-array/array.fbs",
        "./flatbuffers/vortex-dtype/dtype.fbs",
        "./flatbuffers/vortex-file/footer.fbs",
        "./flatbuffers/vortex-layout/layout.fbs",
        "./flatbuffers/vortex-serde/message.fbs",
    ];

    // CD to vortex-flatbuffers project
    sh.change_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../vortex-flatbuffers"));

    cmd!(
        sh,
        "{FLATC_BIN} --rust --filename-suffix '' -I ./flatbuffers/ -o ./src/generated {files...}"
    )
    .run()?;

    Ok(())
}
