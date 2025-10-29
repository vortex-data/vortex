// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use vortex::file::{VortexOpenOptions, VortexWriteOptions};

#[tokio::main]
pub async fn main() -> anyhow::Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let file_name = args.iter().nth(1).expect("must provide file name");
    let path = PathBuf::from(file_name);

    let array_stream = VortexOpenOptions::new()
        .open(path)
        .await?
        .scan()?
        .into_array_stream()?;

    // Pre-allocate a 1GB buffer for the output file.
    let memory_writer = Vec::with_capacity(1024 * 1024 * 1024);
    let writer = VortexWriteOptions::default()
        .write(memory_writer, array_stream)
        .await?;

    println!("Wrote {} bytes to new file", writer.size());

    Ok(())
}
