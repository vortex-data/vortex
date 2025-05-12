use std::fs::File;
use std::path::PathBuf;

use anyhow::Context;
use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use clap::Parser;
use vortex_array::arrow::IntoArrowArray;
use vortex_file::VortexOpenOptions;

#[derive(Parser, Debug)]
pub struct Args {
    /// Path to a Vortex file with geometry data
    input: PathBuf,
}

#[tokio::main]
pub async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let reader = VortexOpenOptions::file()
        .open(&args.input)
        .await
        .context("open input file")?;

    let arrow_schema = reader
        .dtype()
        .to_arrow_schema()
        .context("get arrow schema")?;

    let out = File::open("ipc.arrow").context("open IPC file")?;
    let mut writer = arrow_ipc::writer::StreamWriter::try_new(out, &arrow_schema)?;

    // Stream batches to read the file as-is.
    let batches = reader.scan().context("scan builder")?.into_array_iter()?;

    for batch in batches {
        let record_batch = batch?.into_arrow_preferred()?;
        let batch = RecordBatch::from(
            record_batch
                .as_struct_opt()
                .ok_or_else(|| anyhow::anyhow!("expected struct"))?
                .clone(),
        );
        writer.write(&batch)?;
    }

    eprintln!("wrote ipc.arrow");

    Ok(())
}
