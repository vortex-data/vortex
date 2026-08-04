// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The Hugging Face Hub, served over [`object_store`]'s HTTP store.
//!
//! A Hub repository is a set of files behind an HTTP endpoint that honours range requests, so it
//! needs no cloud SDK of its own: a [`object_store::http::HttpStore`] rooted at the repository's
//! `resolve` prefix, carrying a bearer token when one is available, is the whole implementation.
//! Reads therefore keep every [`object_store::ClientOptions`] setting a caller passes —
//! connect/request timeouts, retries, proxy configuration, `allow_http` — unlike the OpenDAL-backed
//! schemes in [`crate::opendal`], whose bridge owns its own HTTP client.
//!
//! # URL grammar
//!
//! ```text
//! hf://datasets/<owner>/<name>[@<revision>][/<path>]
//! hf://spaces/<owner>/<name>[@<revision>][/<path>]
//! hf://<owner>/<name>[@<revision>][/<path>]              # a model repository
//! ```
//!
//! matching the paths [`huggingface_hub`'s `HfFileSystem`][hffs] accepts. A revision containing `/`
//! (e.g. `refs/convert/parquet`) must be percent-encoded in the URL, as it must be there:
//!
//! ```text
//! hf://datasets/org/name@refs%2Fconvert%2Fparquet/data/train.vortex
//! ```
//!
//! [hffs]: https://huggingface.co/docs/huggingface_hub/guides/hf_file_system
//!
//! # Configuration
//!
//! * `HF_TOKEN` — the API token. Falling back, as `huggingface_hub` does, to the token file at
//!   `HF_TOKEN_PATH`, then `$HF_HOME/token`, then `$HOME/.cache/huggingface/token`. Without a token
//!   the store reads anonymously, which is all a public repository needs.
//! * `HF_ENDPOINT` — the Hub endpoint, defaulting to `https://huggingface.co`.
//!
//! # Limitations
//!
//! The Hub does not implement WebDAV `PROPFIND`, which is how [`object_store`]'s HTTP store lists a
//! prefix, so [`object_store::ObjectStore::list`] fails against these URLs. Opening a file by name
//! works — that is `head` plus ranged `get`, both plain HTTP — so this serves reads of a known path.
//! Callers that need to expand a glob must list the repository through the Hub's own API first and
//! then open each returned path.

use std::sync::Arc;

use http::HeaderMap;
use http::HeaderValue;
use http::header::AUTHORIZATION;
use object_store::ClientOptions;
use object_store::ObjectStore;
use object_store::http::HttpBuilder;
use object_store::path::Path;
use percent_encoding::AsciiSet;
use percent_encoding::CONTROLS;
use percent_encoding::percent_decode_str;
use percent_encoding::utf8_percent_encode;
use url::Url;

/// The URL scheme served by the Hugging Face Hub.
pub const HF_SCHEME: &str = "hf";

/// The Hub endpoint used when `HF_ENDPOINT` is unset.
const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

/// The revision used when a URL does not name one.
const DEFAULT_REVISION: &str = "main";

const ENDPOINT_VAR: &str = "HF_ENDPOINT";
const TOKEN_VAR: &str = "HF_TOKEN";
const TOKEN_PATH_VAR: &str = "HF_TOKEN_PATH";
const HF_HOME_VAR: &str = "HF_HOME";
const HOME_VAR: &str = "HOME";
const ALLOW_HTTP_VAR: &str = "ALLOW_HTTP";

/// The URL authority that marks a dataset repository.
const DATASETS_HOST: &str = "datasets";
/// The URL authority that marks a Space repository.
const SPACES_HOST: &str = "spaces";

/// Characters escaped when a revision is spliced into the `resolve` path.
///
/// `/` is the point of this: the Hub routes `refs/convert/parquet` only in its `%2F` form, since an
/// unescaped `/` would read as another path segment.
const REVISION_ESCAPES: &AsciiSet = &CONTROLS.add(b'/').add(b'%').add(b'?').add(b'#').add(b' ');

/// Whether `scheme` is served by this module.
///
/// Callers dispatching on a URL scheme should ask this rather than comparing against [`HF_SCHEME`]
/// themselves, matching how [`crate::opendal::supports_scheme`] is used.
pub fn supports_scheme(scheme: &str) -> bool {
    scheme == HF_SCHEME
}

/// Which kind of Hub repository a URL addresses.
///
/// The Hub routes each kind under its own path prefix, which is the only thing that differs between
/// them here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HfRepoType {
    /// A dataset repository, `hf://datasets/<owner>/<name>`.
    Dataset,
    /// A model repository, `hf://<owner>/<name>`.
    Model,
    /// A Space repository, `hf://spaces/<owner>/<name>`.
    Space,
}

impl HfRepoType {
    /// The path prefix, including its trailing `/`, that the Hub routes this kind under.
    fn url_prefix(self) -> &'static str {
        match self {
            HfRepoType::Dataset => "datasets/",
            HfRepoType::Model => "",
            HfRepoType::Space => "spaces/",
        }
    }
}

impl std::str::FromStr for HfRepoType {
    type Err = HfStoreError;

    /// Parses either the singular name or the plural the Hub uses in its URLs, so a caller can pass
    /// whichever it has.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "dataset" | "datasets" => Ok(HfRepoType::Dataset),
            "model" | "models" => Ok(HfRepoType::Model),
            "space" | "spaces" => Ok(HfRepoType::Space),
            other => Err(HfStoreError::UnknownRepoType(other.to_string())),
        }
    }
}

/// Error type for building a Hugging Face Hub object store.
#[derive(Debug)]
pub enum HfStoreError {
    /// The URL scheme is not one this module handles.
    UnsupportedScheme(String),
    /// The URL is not a well-formed `hf://` URL.
    InvalidUrl(String),
    /// The repository kind is not one the Hub serves.
    UnknownRepoType(String),
    /// The bearer token could not be used as an HTTP header value.
    InvalidToken(http::header::InvalidHeaderValue),
    /// The underlying HTTP store rejected the configuration.
    Build(object_store::Error),
}

impl std::fmt::Display for HfStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HfStoreError::UnsupportedScheme(s) => write!(f, "unsupported Hugging Face scheme: {s}"),
            HfStoreError::InvalidUrl(url) => write!(
                f,
                "invalid Hugging Face URL {url}: expected hf://datasets/<owner>/<name>[@revision][/path], \
                 hf://spaces/<owner>/<name>[@revision][/path] or hf://<owner>/<name>[@revision][/path]"
            ),
            HfStoreError::UnknownRepoType(kind) => write!(
                f,
                "unknown Hugging Face repository type {kind}: expected one of dataset, model, space"
            ),
            HfStoreError::InvalidToken(e) => write!(f, "invalid Hugging Face token: {e}"),
            HfStoreError::Build(e) => write!(f, "failed to build Hugging Face store: {e}"),
        }
    }
}

impl std::error::Error for HfStoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            HfStoreError::InvalidToken(e) => Some(e),
            HfStoreError::Build(e) => Some(e),
            HfStoreError::UnsupportedScheme(_)
            | HfStoreError::InvalidUrl(_)
            | HfStoreError::UnknownRepoType(_) => None,
        }
    }
}

impl From<HfStoreError> for object_store::Error {
    fn from(e: HfStoreError) -> Self {
        object_store::Error::Generic {
            store: "HuggingFace",
            source: Box::new(e),
        }
    }
}

/// Configuration for a store serving one Hub repository at one revision.
///
/// A store covers a single `(repository, revision)` pair because that pair is what the Hub's
/// `resolve` prefix names; the object key within the store is then the in-repository path.
#[derive(Debug, Clone)]
pub struct HfConfig {
    /// Which kind of repository to address.
    pub repo_type: HfRepoType,
    /// The repository, as `<owner>/<name>`.
    pub repo_id: String,
    /// The revision to read: a branch, tag or commit. Held decoded, so `refs/convert/parquet` is
    /// spelled with `/`; it is percent-encoded when the `resolve` URL is built.
    pub revision: String,
    /// Bearer token for private and gated repositories. `None` reads anonymously.
    pub token: Option<String>,
    /// The Hub endpoint, e.g. `https://huggingface.co`.
    pub endpoint: String,
    /// HTTP client configuration, applied to every read.
    pub client_options: ClientOptions,
}

impl Default for HfConfig {
    fn default() -> Self {
        Self {
            repo_type: HfRepoType::Dataset,
            repo_id: String::new(),
            revision: DEFAULT_REVISION.to_string(),
            token: None,
            endpoint: DEFAULT_ENDPOINT.to_string(),
            client_options: ClientOptions::default(),
        }
    }
}

impl HfConfig {
    /// Configuration for one repository, taking the token and endpoint from the process environment
    /// exactly as resolving an `hf://` URL does.
    ///
    /// `revision` defaults to `main`. Callers that must read anonymously should clear
    /// [`HfConfig::token`] afterwards, since this picks up whatever credentials the environment
    /// offers.
    pub fn from_env(
        repo_type: HfRepoType,
        repo_id: impl Into<String>,
        revision: Option<String>,
    ) -> Self {
        Self::from_env_lookup(repo_type, repo_id, revision, |key| std::env::var(key).ok())
    }

    /// [`HfConfig::from_env`], reading environment configuration through `env_lookup`.
    fn from_env_lookup<F>(
        repo_type: HfRepoType,
        repo_id: impl Into<String>,
        revision: Option<String>,
        env_lookup: F,
    ) -> Self
    where
        F: Fn(&str) -> Option<String>,
    {
        Self {
            repo_type,
            repo_id: repo_id.into(),
            revision: revision.unwrap_or_else(|| DEFAULT_REVISION.to_string()),
            token: resolve_token(&env_lookup),
            endpoint: env_lookup(ENDPOINT_VAR)
                .filter(|endpoint| !endpoint.is_empty())
                .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string()),
            // Every other scheme picks `allow_http` out of the environment, because that is how
            // `parse_url_opts` reads its configuration. Honour it here too, so a plain-HTTP
            // `HF_ENDPOINT` (a test double, or a self-hosted Hub) behaves the same way.
            client_options: ClientOptions::default().with_allow_http(allow_http(&env_lookup)),
        }
    }

    /// The URL this repository's files hang off, which is where the store is rooted.
    fn resolve_prefix(&self) -> String {
        let endpoint = self.endpoint.trim_end_matches('/');
        let prefix = self.repo_type.url_prefix();
        let revision = utf8_percent_encode(&self.revision, REVISION_ESCAPES);
        format!("{endpoint}/{prefix}{}/resolve/{revision}", self.repo_id)
    }
}

/// Build an [`ObjectStore`] for a Hugging Face Hub repository from an [`HfConfig`].
///
/// This is the entry point for callers holding a strongly-typed configuration. The returned store is
/// rooted at the repository's `resolve` prefix, so its object keys are in-repository paths.
pub fn make_hf_store(config: HfConfig) -> Result<Arc<dyn ObjectStore>, HfStoreError> {
    if config.repo_id.is_empty() {
        return Err(HfStoreError::InvalidUrl("<missing repository>".to_string()));
    }

    let mut client_options = config.client_options.clone();
    if let Some(token) = config.token.as_deref() {
        let mut headers = HeaderMap::new();
        let value = HeaderValue::from_str(&format!("Bearer {token}"))
            .map_err(HfStoreError::InvalidToken)?;
        // Sending the token as a default header, rather than per request, is what lets a plain
        // `HttpStore` serve gated and private repositories.
        headers.insert(AUTHORIZATION, value);
        client_options = client_options.with_default_headers(headers);
    }

    let store = HttpBuilder::new()
        .with_url(config.resolve_prefix())
        .with_client_options(client_options)
        .build()
        .map_err(HfStoreError::Build)?;

    Ok(Arc::new(store))
}

/// Build an [`ObjectStore`] for an `hf://` URL, with the in-repository path of the addressed file.
///
/// Configuration not carried by the URL is read from the process environment. The path is reported
/// the way [`object_store::parse_url_opts`] reports it — the key the returned store will see — so
/// that the caller can derive how deep into the URL the store is mounted.
pub fn make_hf_store_from_url(url: &Url) -> Result<(Arc<dyn ObjectStore>, Path), HfStoreError> {
    make_hf_store_from_url_with_env(url, |key| std::env::var(key).ok())
}

/// [`make_hf_store_from_url`], reading environment configuration through `env_lookup`.
///
/// Tests pass a closure over a fixed map so they do not race against the process environment.
pub(crate) fn make_hf_store_from_url_with_env<F>(
    url: &Url,
    env_lookup: F,
) -> Result<(Arc<dyn ObjectStore>, Path), HfStoreError>
where
    F: Fn(&str) -> Option<String>,
{
    let (config, path) = url_to_config(url, env_lookup)?;
    Ok((make_hf_store(config)?, path))
}

/// Translate an `hf://` URL into an [`HfConfig`] plus the in-repository path it addresses.
fn url_to_config<F>(url: &Url, env_lookup: F) -> Result<(HfConfig, Path), HfStoreError>
where
    F: Fn(&str) -> Option<String>,
{
    if !supports_scheme(url.scheme()) {
        return Err(HfStoreError::UnsupportedScheme(url.scheme().to_string()));
    }
    let invalid = || HfStoreError::InvalidUrl(url.to_string());

    // `hf://datasets/org/name/...` parses with `datasets` as the authority and `/org/name/...` as
    // the path, so the repository kind is the authority and a model repository's owner is too.
    let host = url.host_str().ok_or_else(invalid)?;
    let segments: Vec<&str> = url.path().split('/').filter(|s| !s.is_empty()).collect();

    let (repo_type, owner, rest) = match host {
        DATASETS_HOST | SPACES_HOST => {
            let repo_type = if host == DATASETS_HOST {
                HfRepoType::Dataset
            } else {
                HfRepoType::Space
            };
            let (owner, rest) = segments.split_first().ok_or_else(invalid)?;
            (repo_type, *owner, rest)
        }
        owner => (HfRepoType::Model, owner, segments.as_slice()),
    };

    let (name, rest) = rest.split_first().ok_or_else(invalid)?;
    if owner.is_empty() || name.is_empty() {
        return Err(invalid());
    }

    // A revision is appended to the repository name with `@`, and arrives percent-encoded when it
    // contains `/`. Hold it decoded; `resolve_prefix` re-encodes it on the way back out.
    let (name, revision) = match name.split_once('@') {
        Some((name, revision)) => {
            if name.is_empty() || revision.is_empty() {
                return Err(invalid());
            }
            let revision = percent_decode_str(revision)
                .decode_utf8()
                .map_err(|_| invalid())?
                .into_owned();
            (name, revision)
        }
        None => (*name, DEFAULT_REVISION.to_string()),
    };

    let config = HfConfig::from_env_lookup(
        repo_type,
        format!("{owner}/{name}"),
        Some(revision),
        &env_lookup,
    );

    // The segments still carry their URL escapes, so decode them into the object key rather than
    // joining them raw.
    let path = Path::from_url_path(rest.join("/")).map_err(|_| invalid())?;

    Ok((config, path))
}

/// Whether plain-HTTP endpoints are permitted, from the `ALLOW_HTTP` variable the `object_store`
/// builders read. Anything other than a `true`-ish value leaves HTTPS as the requirement.
fn allow_http<F>(env_lookup: &F) -> bool
where
    F: Fn(&str) -> Option<String>,
{
    env_lookup(ALLOW_HTTP_VAR)
        .is_some_and(|value| matches!(value.trim().to_ascii_lowercase().as_str(), "true" | "1"))
}

/// The bearer token to read with, following the same precedence as `huggingface_hub.get_token()`:
/// `HF_TOKEN`, then the token file at `HF_TOKEN_PATH`, `$HF_HOME/token` or
/// `$HOME/.cache/huggingface/token`.
fn resolve_token<F>(env_lookup: &F) -> Option<String>
where
    F: Fn(&str) -> Option<String>,
{
    if let Some(token) = env_lookup(TOKEN_VAR).filter(|token| !token.is_empty()) {
        return Some(token);
    }

    let token_path = env_lookup(TOKEN_PATH_VAR)
        .or_else(|| env_lookup(HF_HOME_VAR).map(|home| format!("{home}/token")))
        .or_else(|| env_lookup(HOME_VAR).map(|home| format!("{home}/.cache/huggingface/token")))?;

    // A missing token file is the ordinary anonymous case, not an error.
    let token = std::fs::read_to_string(token_path).ok()?;
    let token = token.trim();
    (!token.is_empty()).then(|| token.to_string())
}

#[cfg(test)]
mod tests;
