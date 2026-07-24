# Install

## `vx` CLI

The `vx` command-line tool lets you convert, inspect, browse, and query Vortex files from
the terminal. You have the following options:

1. Binstall (recommended) downloads a pre-built binary. Requires
[cargo-binstall](https://github.com/cargo-bins/cargo-binstall):

```bash
cargo binstall vortex-tui
```

2. pip also installs the Python library; see the [Python quickstart](python.rst) for library usage:

```bash
pip install vortex-data
```

3. uvx runs the CLI without installing. Requires [uv](https://docs.astral.sh/uv/):

```bash
uvx --from vortex-data vx --help
```

4. Cargo builds from source, which can be slow due to the large dependency tree:

```bash
cargo install vortex-tui
```

Verify the installation:

```bash
vx --help
```

## Using Vortex as a library

The `vx` CLI is the quickest way to get started, but Vortex can also be used as a library
for reading, writing, and manipulating compressed arrays programmatically. See the language
quickstarts for [Python](python.rst), [Rust](rust.rst), and [Java](java.md).

## Sample data

This quickstart uses the NYC Yellow Taxi trip dataset. More months and datasets are available
from the [NYC TLC trip record data page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

Download a single month (~50 MB Parquet):

```bash
curl -O https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet
```
