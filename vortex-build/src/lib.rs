// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Build script helpers for compiling the FlatBuffers and Protocol Buffers schemas that ship with
//! the Vortex crates.
//!
//! Every schema lives in the crate that owns the types it describes and is compiled into `OUT_DIR`
//! by that crate's build script. Schemas referencing definitions owned by another crate name it
//! explicitly, so nothing is discovered by walking the workspace and a path dependency behaves the
//! same as a package unpacked from a registry:
//!
//! ```rust,ignore
//! vortex_build::flatbuffers()
//!     .depends_on("vortex-array")
//!     .compile(&["vortex-serde/message.fbs"]);
//! ```

#![deny(missing_docs)]
// Build scripts have no error channel back to Cargo, so failures are reported by panicking.
#![allow(clippy::expect_used)]
#![allow(clippy::manual_assert)]
#![allow(clippy::panic)]

use std::env;
use std::fs::create_dir_all;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use prost_build::Config;

const FLATBUFFERS_KEY: &str = "flatbuffers";
const PROTO_KEY: &str = "proto";

/// Module path, relative to the crate root, where generated FlatBuffers code looks for the schemas
/// it includes from other crates. Crates with cross-crate includes define it by hand; see
/// `vortex-ipc/src/flatbuffers.rs`.
const FLATBUFFERS_INCLUDE_PREFIX: &str = "flatbuffers::deps";

/// Compiles this crate's FlatBuffers schemas from `flatbuffers/` into `$OUT_DIR/flatbuffers`.
pub fn flatbuffers() -> FlatBuffers {
    let schema_dir = manifest_dir().join(FLATBUFFERS_KEY);
    FlatBuffers {
        includes: vec![schema_dir.clone()],
        schema_dir,
    }
}

/// Compiles this crate's Protocol Buffers schemas from `proto/` into `$OUT_DIR/proto`.
pub fn proto() -> Proto {
    let schema_dir = manifest_dir().join(PROTO_KEY);
    Proto {
        includes: vec![schema_dir.clone()],
        schema_dir,
    }
}

/// Builder for FlatBuffers compilation, driven by `flatc` from `FLATC` or `PATH`.
pub struct FlatBuffers {
    schema_dir: PathBuf,
    includes: Vec<PathBuf>,
}

impl FlatBuffers {
    /// Makes the FlatBuffers schemas of the direct dependency declaring `links = "<links>"`
    /// available to `include` statements.
    #[must_use]
    pub fn depends_on(mut self, links: &str) -> Self {
        self.includes.push(dep_schema_dir(links, FLATBUFFERS_KEY));
        self
    }

    /// Compiles the given schemas, each named relative to this crate's `flatbuffers` directory.
    pub fn compile(self, schemas: &[&str]) {
        let out_dir = out_dir().join(FLATBUFFERS_KEY);
        create_dir_all(&out_dir)
            .unwrap_or_else(|e| panic!("failed to create {}: {e}", out_dir.display()));

        let mut flatc = Command::new(flatc_binary());
        flatc
            .arg("--rust")
            // Vortex modules are named for the schema, so drop flatc's `_generated` suffix.
            .args(["--filename-suffix", ""])
            .args(["--include-prefix", FLATBUFFERS_INCLUDE_PREFIX])
            .arg("-o")
            .arg(&out_dir);

        for include in &self.includes {
            rerun_if_changed(include);
            flatc.arg("-I").arg(include);
        }

        for schema in schemas {
            let path = self.schema_dir.join(schema);
            assert!(path.exists(), "schema not found: {}", path.display());
            flatc.arg(path);
        }

        run(flatc);
        export_schema_dir(FLATBUFFERS_KEY, &self.schema_dir);
    }
}

/// Builder for Protocol Buffers compilation. Parsing uses [`protox`], so `protoc` is not needed.
pub struct Proto {
    schema_dir: PathBuf,
    includes: Vec<PathBuf>,
}

impl Proto {
    /// Makes the Protocol Buffers schemas of the direct dependency declaring `links = "<links>"`
    /// available to `import` statements.
    #[must_use]
    pub fn depends_on(mut self, links: &str) -> Self {
        self.includes.push(dep_schema_dir(links, PROTO_KEY));
        self
    }

    /// Compiles the given schemas, each named relative to this crate's `proto` directory.
    pub fn compile(self, schemas: &[&str]) {
        let out_dir = out_dir().join(PROTO_KEY);
        create_dir_all(&out_dir)
            .unwrap_or_else(|e| panic!("failed to create {}: {e}", out_dir.display()));

        for include in &self.includes {
            rerun_if_changed(include);
        }

        let file_descriptors = protox::compile(schemas, &self.includes)
            .unwrap_or_else(|e| panic!("failed to compile protos: {e}"));

        Config::new()
            .out_dir(&out_dir)
            .compile_fds(file_descriptors)
            .unwrap_or_else(|e| panic!("failed to generate proto bindings: {e}"));

        export_schema_dir(PROTO_KEY, &self.schema_dir);
    }
}

/// Publishes `dir` to direct dependents as `DEP_<LINKS>_<KEY>`. Cargo only forwards build script
/// metadata for packages declaring `links`, so this is a no-op for the others.
fn export_schema_dir(key: &str, dir: &Path) {
    if env::var_os("CARGO_MANIFEST_LINKS").is_some() {
        println!("cargo::metadata={key}={}", dir.display());
    }
}

fn dep_schema_dir(links: &str, key: &str) -> PathBuf {
    let var = format!("DEP_{}_{}", env_fragment(links), env_fragment(key));
    let dir = env::var_os(&var).unwrap_or_else(|| {
        panic!(
            "{var} is not set: `{links}` must be a direct dependency of this crate and must \
             declare `links = \"{links}\"`"
        )
    });
    PathBuf::from(dir)
}

/// Spells `value` the way Cargo spells the components of its `DEP_*` variables.
fn env_fragment(value: &str) -> String {
    value
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>()
        .to_uppercase()
}

fn flatc_binary() -> PathBuf {
    println!("cargo::rerun-if-env-changed=FLATC");
    env::var_os("FLATC").map_or_else(|| PathBuf::from("flatc"), PathBuf::from)
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is not set"))
}

fn out_dir() -> PathBuf {
    PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is not set"))
}

fn rerun_if_changed(path: &Path) {
    println!("cargo::rerun-if-changed={}", path.display());
}

fn run(mut command: Command) {
    let program = command.get_program().to_string_lossy().into_owned();
    let status = command.status().unwrap_or_else(|e| {
        panic!(
            "failed to run {program}: {e}. Install the FlatBuffers compiler, or set FLATC to its \
             location."
        )
    });
    assert!(status.success(), "{program} failed with {status}");
}
