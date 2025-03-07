# DuckDB Rust extension template
This is an **experimental** template for Rust based extensions based on the C Extension API of DuckDB. The goal is to 
turn this eventually into a stable basis for pure-Rust DuckDB extensions that can be submitted to the Community extensions
repository

Features:
- No DuckDB build required
- No C++ or C code required
- CI/CD chain preconfigured
- (Coming soon) Works with community extensions

## Cloning

Clone the repo with submodules

```shell
git clone --recurse-submodules <repo>
```

## Dependencies
In principle, these extensions can be compiled with the Rust toolchain alone. However, this template relies on some additional
tooling to make life a little easier and to be able to share CI/CD infrastructure with extension templates for other languages:

- Python3
- Python3-venv
- [Make](https://www.gnu.org/software/make)
- Git

Installing these dependencies will vary per platform:
- For Linux, these come generally pre-installed or are available through the distro-specific package manager.
- For MacOS, [homebrew](https://formulae.brew.sh/).
- For Windows, [chocolatey](https://community.chocolatey.org/).

## Building
After installing the dependencies, building is a two-step process. Firstly run:
```shell
make configure
```
This will ensure a Python venv is set up with DuckDB and DuckDB's test runner installed. Additionally, depending on configuration,
DuckDB will be used to determine the correct platform for which you are compiling.

Then, to build the extension run:
```shell
make debug
```
This delegates the build process to cargo, which will produce a shared library in `target/debug/<shared_lib_name>`. After this step, 
a script is run to transform the shared library into a loadable extension by appending a binary footer. The resulting extension is written
to the `build/debug` directory.

To create optimized release binaries, simply run `make release` instead.

## Testing
This extension uses the DuckDB Python client for testing. This should be automatically installed in the `make configure` step.
The tests themselves are written in the SQLLogicTest format, just like most of DuckDB's tests. A sample test can be found in
`test/sql/<extension_name>.test`. To run the tests using the *debug* build:

```shell
make test_debug
```

or for the *release* build:
```shell
make test_release
```

### Version switching 
Testing with different DuckDB versions is really simple:

First, run 
```
make clean_all
```
to ensure the previous `make configure` step is deleted.

Then, run 
```
DUCKDB_TEST_VERSION=v1.1.2 make configure
```
to select a different duckdb version to test with

Finally, build and test with 
```
make debug
make test_debug
```

### Known issues
This is a bit of a footgun, but the extensions produced by this template may (or may not) be broken on windows on python3.11 
with the following error on extension load:
```shell
IO Error: Extension '<name>.duckdb_extension' could not be loaded: The specified module could not be found
```
This was resolved by using python 3.12