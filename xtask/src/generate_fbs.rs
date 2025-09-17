// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::{Path, PathBuf};

use xshell::{Shell, cmd};

// static FLATC_BIN: &str = "flatc";
static PLANUS_BIN: &str = "/Users/adamgs/.cargo/bin/planus";

pub fn generate_fbs() -> anyhow::Result<()> {
    let sh = Shell::new()?;

    let files = vec![
        "flatbuffers/vortex-array/array.fbs",
        "flatbuffers/vortex-dtype/dtype.fbs",
        "flatbuffers/vortex-file/footer.fbs",
        "flatbuffers/vortex-layout/layout.fbs",
        "flatbuffers/vortex-serde/message.fbs",
    ];

    // CD to vortex-flatbuffers project
    sh.change_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../vortex-flatbuffers"));

    // cmd!(
    //     sh,
    //     "{FLATC_BIN} --rust --filename-suffix '' -I ./flatbuffers/ -o ./src/generated {files...}"
    // )
    // .run()?;

    let base = Path::new("./src/generated/");
    let cwd = sh.current_dir();
    println!("CWD: {}", cwd.display());
    println!("Base: {}", base.display());

    for file in files {
        // let file = Path::new(file).canonicalize().unwrap();
        println!("File is: {file}");
        let output_file = Path::new(file).with_extension("rs");
        let out_file_only = output_file.file_name().unwrap();
        let output = base.join(out_file_only);

        let input_path = Path::new(file).to_string_lossy().to_string();
        let output_path = output.to_string_lossy().to_string();
        println!("Reading: {input_path}");
        println!("Writing to: {output_path}");

        cmd!(sh, "{PLANUS_BIN} rust -o {output_path} {input_path}").run()?;
    }

    Ok(())
}
