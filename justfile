data_input := "/Users/connor/spiral/hw25/output/transformed_benchmarks.json"
data_output := "/Users/connor/spiral/vortex-data/vortex/vortex-wasm/data.vortex"
commits_input := "/Users/connor/spiral/hw25/output/transformed_commits.json"
commits_output := "/Users/connor/spiral/vortex-data/vortex/vortex-wasm/commits.vortex"

# Migrate benchmark data from JSON to Vortex format.
migrate-data:
    cargo run -p vortex-wasm --bin migrate_data --release -- {{data_input}} {{data_output}}

# Migrate commits from JSON to Vortex format.
migrate-commits:
    cargo run -p vortex-wasm --bin migrate_commits --release -- {{commits_input}} {{commits_output}}

# Run both migrations.
migrate-all: migrate-data migrate-commits

browse file:
    cargo run -p vortex-tui -- browse {{file}}

build-wasm:
    cargo build -p vortex-wasm --no-default-features --lib --target wasm32-unknown-unknown --release
