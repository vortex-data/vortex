use clap::Parser;
use xshell::{Shell, cmd};

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

fn main() -> anyhow::Result<()> {
    let cli = Xtask::parse();
    match cli.command {
        Commands::GenerateFlatbuffers => execute_generate_fbs()?,
        Commands::GenerateProto => execute_generate_proto()?,
    }
    Ok(())
}
