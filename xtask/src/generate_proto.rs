// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

pub fn generate_proto() -> anyhow::Result<()> {
    let vortex_proto = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../vortex-proto");
    let proto_files = vec![
        vortex_proto.join("proto").join("dtype.proto"),
        vortex_proto.join("proto").join("scalar.proto"),
        vortex_proto.join("proto").join("expr.proto"),
    ];

    for file in &proto_files {
        if !file.exists() {
            anyhow::bail!("proto file not found: {file:?}");
        }
    }

    let out_dir = vortex_proto.join("src").join("generated");
    std::fs::create_dir_all(&out_dir)?;

    prost_build::Config::new()
        .out_dir(out_dir)
        .compile_protos(&proto_files, &[vortex_proto.join("proto")])?;

    Ok(())
}
