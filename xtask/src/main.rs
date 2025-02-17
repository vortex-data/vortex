use std::path::PathBuf;

use bench_vortex::tpch::duckdb::{generate_tpch, DuckdbTpchOptions};
use bench_vortex::tpch::load_datasets;
use bench_vortex::Format;
use clap::Parser;
use tokio::runtime::Builder;
use url::Url;
use xshell::{cmd, Shell};

static FLATC_BIN: &str = "flatc";

#[derive(clap::Parser)]
struct Xtask {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    #[command(name = "generate-fbs")]
    GenerateFlatbuffers,
    #[command(name = "generate-proto")]
    GenerateProto,
    #[command(name = "generate-tpch-csvs")]
    GenerateTpchCsvs {
        scale_factor: Option<u8>,
        output_dir: Option<PathBuf>,
    },
    #[command(name = "tpch-csv-to-parquet")]
    TpchCsvToParquet { base_dir: Option<PathBuf> },
    #[command(name = "tpch-csv-to-vortex")]
    TpchCsvToVortex { base_dir: Option<PathBuf> },
}

fn execute_generate_fbs() -> anyhow::Result<()> {
    let sh = Shell::new()?;

    let files = vec![
        "./flatbuffers/vortex-dtype/dtype.fbs",
        "./flatbuffers/vortex-scalar/scalar.fbs",
        "./flatbuffers/vortex-array/array.fbs",
        "./flatbuffers/vortex-file/footer.fbs",
        "./flatbuffers/vortex-layout/layout.fbs",
        "./flatbuffers/vortex-serde/message.fbs",
    ];

    // CD to vortex-flatbuffers project
    sh.change_dir(std::env::current_dir()?.join("vortex-flatbuffers"));

    cmd!(
        sh,
        "{FLATC_BIN} --rust --filename-suffix '' -I ./flatbuffers/ -o ./src/generated {files...}"
    )
    .run()?;

    Ok(())
}

fn execute_generate_proto() -> anyhow::Result<()> {
    let vortex_proto = std::env::current_dir()?.join("vortex-proto");
    let proto_files = vec![
        vortex_proto.join("proto").join("dtype.proto"),
        vortex_proto.join("proto").join("scalar.proto"),
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

fn execute_generate_tpch_csv(
    scale_factor: Option<u8>,
    base_dir: Option<PathBuf>,
) -> anyhow::Result<()> {
    let default = DuckdbTpchOptions::default();
    let conf = DuckdbTpchOptions {
        scale_factor: scale_factor.unwrap_or(default.scale_factor),
        base_dir: base_dir.unwrap_or(default.base_dir),
    };
    generate_tpch(conf)?;
    Ok(())
}

fn execute_from_tpch_csv(base_dir: Option<PathBuf>, format: Format) -> anyhow::Result<()> {
    let runtime = Builder::new_multi_thread().enable_all().build()?;
    let base_dir = base_dir.unwrap_or_else(|| DuckdbTpchOptions::default().csvs_dir());
    // add a trailing slash to bse_dir so path concat works as expected
    let base_url = Url::parse(
        ("file:".to_owned()
            + base_dir
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("must be utf8"))?
            + "/")
            .as_ref(),
    )?;
    runtime.block_on(load_datasets(&base_url, format, false))?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = Xtask::parse();
    match cli.command {
        Commands::GenerateFlatbuffers => execute_generate_fbs()?,
        Commands::GenerateProto => execute_generate_proto()?,
        Commands::GenerateTpchCsvs {
            scale_factor,
            output_dir,
        } => execute_generate_tpch_csv(scale_factor, output_dir)?,
        Commands::TpchCsvToParquet { base_dir } => {
            execute_from_tpch_csv(base_dir, Format::Parquet)?
        }
        Commands::TpchCsvToVortex { base_dir } => {
            execute_from_tpch_csv(base_dir, Format::OnDiskVortex)?
        }
    }
    Ok(())
}
