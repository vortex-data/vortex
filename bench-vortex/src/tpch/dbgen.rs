/// Download TPC-H data via Docker.
///
/// This is a simple wrapper around the `dbgen` tool.
///
/// For more information, see the [dbgen](https://github.com/spiraldb/tpch-dbgen) project.
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::copy;
use std::path::{Path, PathBuf};
use std::process::Command;

use itertools::Itertools;
use tar::Archive;
use xshell::Shell;

pub struct DBGen {
    options: DBGenOptions,
}

pub struct DBGenOptions {
    /// Scale factor of the data in GB.
    pub scale_factor: u8,

    /// Location on-disk to store generated files.
    pub base_dir: PathBuf,

    /// Location of where we may cache the dbgen tool download.
    pub cache_dir: PathBuf,
}

impl Default for DBGenOptions {
    /// Generate a default options.
    ///
    /// # Panics
    ///
    /// Will panic if `std::env::current_dir()` fails with error.
    fn default() -> Self {
        Self {
            scale_factor: 1,
            base_dir: std::env::current_dir().unwrap().join("data").join("tpch"),
            cache_dir: homedir::my_home()
                .unwrap()
                .unwrap()
                .join(".cache")
                .join("vortex")
                .join("dbgen"),
        }
    }
}

impl DBGenOptions {
    pub fn with_base_dir<P: AsRef<Path>>(self, dir: P) -> Self {
        Self {
            base_dir: dir.as_ref().to_owned(),
            scale_factor: self.scale_factor,
            cache_dir: self.cache_dir,
        }
    }
}

impl DBGen {
    pub fn new(options: DBGenOptions) -> Self {
        Self { options }
    }
}

impl DBGen {
    #[allow(clippy::unwrap_in_result)]
    /// Generate the TPC-H data files for use with benchmarks.
    pub fn generate(&self) -> anyhow::Result<PathBuf> {
        let sh = Shell::new()?;

        let scale_factor = self.options.scale_factor.to_string();

        // mkdir -p the output directory
        let output_dir = self.options.base_dir.join(scale_factor.as_str());
        sh.create_dir(&output_dir)?;

        // See if the success file has been written. If so, do not run expensive generator
        // process again.
        let success_file = output_dir.join(".success");
        if sh.path_exists(&success_file) {
            return Ok(output_dir);
        }

        let dbgen_binary = get_cached_dbgen(&self.options.cache_dir)?;
        let dists_file = dbgen_binary.parent().unwrap().join("dists.dss");

        // Generate the files using our DBGen tool
        let output = Command::new(dbgen_binary)
            .current_dir(&output_dir)
            .args(vec![
                "-b",
                dists_file.into_os_string().into_string().unwrap().as_str(),
                "-s",
                scale_factor.as_str(),
                "-f",
                "-v",
            ])
            .output()?;

        if !output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("dbgen failed: stdout=\"{stdout}\", stderr=\"{stderr}\"");
        }

        // Every tpch .tbl file is a pipe-separated values, but for some strange reason, *also* includes
        // a trailing pipe at the end of every line.
        // DataFusion's CSV reader (and assumedly most others) do not support this, so we rewrite all the
        // files before completing generation.
        for table_path in sh.read_dir(&output_dir)? {
            clean_trailing_pipes(&sh, &table_path)?;
        }

        // Write a success file to indicate this scale-factor is created.
        sh.write_file(success_file, vec![])?;

        Ok(output_dir)
    }
}

fn clean_trailing_pipes<P: AsRef<Path>>(sh: &Shell, path: P) -> anyhow::Result<()> {
    let stripped = sh
        .read_file(&path)?
        .lines()
        .map(|line| line.strip_suffix("|").unwrap())
        .join("\n");
    sh.write_file(&path, stripped.as_bytes())?;

    Ok(())
}

#[derive(Clone, Copy)]
enum Platform {
    MacOS,
    Linux,
}

impl Display for Platform {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            Platform::MacOS => "macos",
            Platform::Linux => "linux",
        };
        write!(f, "{}", str)
    }
}

// Increment this when we release new tpch-dbgen.
const DBGEN_VERSION: &str = "0.1.0";

// Return a handle to the downloaded toolchain.
fn get_cached_dbgen<P: AsRef<Path>>(cache_dir: P) -> anyhow::Result<PathBuf> {
    if cfg!(target_os = "macos") {
        return get_or_cache_toolchain(cache_dir.as_ref(), DBGEN_VERSION, Platform::MacOS);
    }

    if cfg!(target_os = "linux") {
        return get_or_cache_toolchain(cache_dir.as_ref(), DBGEN_VERSION, Platform::Linux);
    }

    anyhow::bail!("unsupported platform, only linux and macos supported")
}

fn get_or_cache_toolchain(
    cache_dir: &Path,
    version: &str,
    platform: Platform,
) -> anyhow::Result<PathBuf> {
    let download_dir = dbgen_dir(cache_dir, version, platform);
    std::fs::create_dir_all(&download_dir)?;

    let url = format!("https://github.com/spiraldb/tpch-dbgen/releases/download/{version}/dbgen-{platform}-{version}.tar");

    let mut zip_file = reqwest::blocking::get(url)?;
    let zip_path = download_dir.join(
        zip_file
            .url()
            .path_segments()
            .and_then(Iterator::last)
            .unwrap(),
    );

    {
        let mut file = File::create(&zip_path)?;
        copy(&mut zip_file, &mut file)?;
    }

    let file = File::open(&zip_path)?;
    let mut archive = Archive::new(file);

    for entry in archive.entries()? {
        let mut entry = entry?;
        if !entry.unpack_in(&download_dir)? {
            anyhow::bail!("failed to extract {:?} in {download_dir:?}", entry.path()?);
        }
    }

    Ok(dbgen_binary(cache_dir, version, platform))
}

fn dbgen_dir(cache_dir: &Path, version: &str, platform: Platform) -> PathBuf {
    cache_dir.join(version).join(platform.to_string())
}

fn dbgen_binary(cache_dir: &Path, version: &str, platform: Platform) -> PathBuf {
    dbgen_dir(cache_dir, version, platform).join("dbgen")
}
