// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared `--store-option key=value` CLI flag for configuring remote object stores.

/// Reusable object store configuration flag.
///
/// Each `key=value` is forwarded to the object store when opening remote URLs (`s3://`, `gs://`,
/// `az://`, ...), taking precedence over the corresponding environment variable. Options are
/// ignored for local files and `file://` URLs.
#[derive(Debug, Clone, Default, clap::Args)]
pub struct StoreOptions {
    /// Object store options as `key=value`. Pass several comma-separated or repeat the flag, e.g.
    /// `--store-option region=us-east-1,allow_http=true` or
    /// `--store-option region=us-east-1 --store-option allow_http=true`.
    ///
    /// Note: a value cannot contain a comma; use a repeated flag for such values.
    #[arg(
        long = "store-option",
        value_name = "KEY=VALUE",
        value_delimiter = ',',
        value_parser = parse_key_val
    )]
    pub options: Vec<(String, String)>,
}

impl StoreOptions {
    /// Returns the options as an owned `Vec`, suitable for `resolve_with_props` /
    /// `open_url_with_props`.
    pub fn props(&self) -> Vec<(String, String)> {
        self.options.clone()
    }
}

/// Parse a single `key=value` store option.
fn parse_key_val(s: &str) -> Result<(String, String), String> {
    let (key, value) = s
        .split_once('=')
        .ok_or_else(|| format!("invalid store option `{s}`, expected `key=value`"))?;
    Ok((key.to_string(), value.to_string()))
}
