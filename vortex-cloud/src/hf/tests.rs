// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use url::Url;

use super::DEFAULT_ENDPOINT;
use super::HfConfig;
use super::HfRepoType;
use super::HfStoreError;
use super::url_to_config;

/// Configuration lookups over a fixed map, so these tests neither read nor mutate the process
/// environment.
fn env(vars: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> + use<> {
    let vars: Vec<(String, String)> = vars
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
    move |key| {
        vars.iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.to_string())
    }
}

/// Parse a URL with no environment configuration at all, so no token is found and the default
/// endpoint applies.
fn parse(url: &str) -> Result<(HfConfig, String), Box<dyn std::error::Error>> {
    let url = Url::parse(url)?;
    let (config, path) = url_to_config(&url, env(&[]))?;
    Ok((config, path.as_ref().to_string()))
}

#[rstest]
// A dataset repository, with and without an explicit revision.
#[case("hf://datasets/org/name", HfRepoType::Dataset, "org/name", "main", "")]
#[case(
    "hf://datasets/org/name@main",
    HfRepoType::Dataset,
    "org/name",
    "main",
    ""
)]
#[case(
    "hf://datasets/org/name/train.vortex",
    HfRepoType::Dataset,
    "org/name",
    "main",
    "train.vortex"
)]
#[case(
    "hf://datasets/org/name/data/nested/train.vortex",
    HfRepoType::Dataset,
    "org/name",
    "main",
    "data/nested/train.vortex"
)]
#[case(
    "hf://datasets/org/name@v1.0/train.vortex",
    HfRepoType::Dataset,
    "org/name",
    "v1.0",
    "train.vortex"
)]
// A revision containing `/` arrives percent-encoded and must be held decoded.
#[case(
    "hf://datasets/org/name@refs%2Fconvert%2Fparquet/data/train.vortex",
    HfRepoType::Dataset,
    "org/name",
    "refs/convert/parquet",
    "data/train.vortex"
)]
// A bare owner/name is a model repository: the owner is the URL authority.
#[case(
    "hf://org/name/model.vortex",
    HfRepoType::Model,
    "org/name",
    "main",
    "model.vortex"
)]
#[case(
    "hf://org/name@abc123/model.vortex",
    HfRepoType::Model,
    "org/name",
    "abc123",
    "model.vortex"
)]
#[case(
    "hf://spaces/org/name/app.vortex",
    HfRepoType::Space,
    "org/name",
    "main",
    "app.vortex"
)]
fn test_url_to_config(
    #[case] url: &str,
    #[case] repo_type: HfRepoType,
    #[case] repo_id: &str,
    #[case] revision: &str,
    #[case] path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let (config, parsed_path) = parse(url)?;

    assert_eq!(config.repo_type, repo_type);
    assert_eq!(config.repo_id, repo_id);
    assert_eq!(config.revision, revision);
    assert_eq!(parsed_path, path);
    Ok(())
}

#[rstest]
// A repository needs both an owner and a name.
#[case("hf://datasets/name-only")]
#[case("hf://org")]
// An `@` must have a name on the left and a revision on the right.
#[case("hf://datasets/org/@main")]
#[case("hf://datasets/org/name@")]
fn test_url_to_config_invalid(#[case] url: &str) -> Result<(), Box<dyn std::error::Error>> {
    let url = Url::parse(url)?;
    assert!(matches!(
        url_to_config(&url, env(&[])),
        Err(HfStoreError::InvalidUrl(_))
    ));
    Ok(())
}

#[test]
fn test_url_to_config_rejects_other_schemes() -> Result<(), Box<dyn std::error::Error>> {
    let url = Url::parse("s3://bucket/key.vortex")?;
    assert!(matches!(
        url_to_config(&url, env(&[])),
        Err(HfStoreError::UnsupportedScheme(_))
    ));
    Ok(())
}

#[rstest]
// Each repository kind hangs off its own prefix, and the store is rooted at the revision.
#[case(
    HfRepoType::Dataset,
    "main",
    "https://huggingface.co/datasets/org/name/resolve/main"
)]
#[case(
    HfRepoType::Model,
    "main",
    "https://huggingface.co/org/name/resolve/main"
)]
#[case(
    HfRepoType::Space,
    "main",
    "https://huggingface.co/spaces/org/name/resolve/main"
)]
// The Hub only routes the escaped form of a revision containing `/`.
#[case(
    HfRepoType::Dataset,
    "refs/convert/parquet",
    "https://huggingface.co/datasets/org/name/resolve/refs%2Fconvert%2Fparquet"
)]
fn test_resolve_prefix(
    #[case] repo_type: HfRepoType,
    #[case] revision: &str,
    #[case] expected: &str,
) {
    let config = HfConfig {
        repo_type,
        repo_id: "org/name".to_string(),
        revision: revision.to_string(),
        ..HfConfig::default()
    };

    assert_eq!(config.resolve_prefix(), expected);
}

/// A percent-encoded revision must survive the URL -> config -> URL round trip, since that is how a
/// `refs/convert/parquet` read reaches the Hub.
#[test]
fn test_slash_revision_round_trips() -> Result<(), Box<dyn std::error::Error>> {
    let (config, path) =
        parse("hf://datasets/org/name@refs%2Fconvert%2Fparquet/data/train.vortex")?;

    assert_eq!(config.revision, "refs/convert/parquet");
    assert_eq!(
        config.resolve_prefix(),
        "https://huggingface.co/datasets/org/name/resolve/refs%2Fconvert%2Fparquet"
    );
    assert_eq!(path, "data/train.vortex");
    Ok(())
}

#[test]
fn test_endpoint_override() -> Result<(), Box<dyn std::error::Error>> {
    let url = Url::parse("hf://datasets/org/name/train.vortex")?;
    let (config, _path) = url_to_config(&url, env(&[("HF_ENDPOINT", "https://hub.example.com/")]))?;

    assert_eq!(config.endpoint, "https://hub.example.com/");
    // The trailing slash on the endpoint must not double up in the resolve prefix.
    assert_eq!(
        config.resolve_prefix(),
        "https://hub.example.com/datasets/org/name/resolve/main"
    );
    Ok(())
}

#[test]
fn test_default_endpoint_when_unset_or_empty() -> Result<(), Box<dyn std::error::Error>> {
    let url = Url::parse("hf://datasets/org/name")?;

    let (unset, _) = url_to_config(&url, env(&[]))?;
    assert_eq!(unset.endpoint, DEFAULT_ENDPOINT);

    let (empty, _) = url_to_config(&url, env(&[("HF_ENDPOINT", "")]))?;
    assert_eq!(empty.endpoint, DEFAULT_ENDPOINT);
    Ok(())
}

#[test]
fn test_token_from_env() -> Result<(), Box<dyn std::error::Error>> {
    let url = Url::parse("hf://datasets/org/name")?;
    let (config, _path) = url_to_config(&url, env(&[("HF_TOKEN", "hf_from_env")]))?;

    assert_eq!(config.token.as_deref(), Some("hf_from_env"));
    Ok(())
}

#[test]
fn test_token_from_token_file() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let token_file = dir.path().join("token");
    // The saved login file is newline-terminated, which must not reach the header value.
    std::fs::write(&token_file, "hf_from_file\n")?;

    let url = Url::parse("hf://datasets/org/name")?;
    let (config, _path) = url_to_config(
        &url,
        env(&[("HF_TOKEN_PATH", &token_file.to_string_lossy())]),
    )?;

    assert_eq!(config.token.as_deref(), Some("hf_from_file"));
    Ok(())
}

#[test]
fn test_token_env_wins_over_file() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let token_file = dir.path().join("token");
    std::fs::write(&token_file, "hf_from_file")?;

    let url = Url::parse("hf://datasets/org/name")?;
    let (config, _path) = url_to_config(
        &url,
        env(&[
            ("HF_TOKEN", "hf_from_env"),
            ("HF_TOKEN_PATH", &token_file.to_string_lossy()),
        ]),
    )?;

    assert_eq!(config.token.as_deref(), Some("hf_from_env"));
    Ok(())
}

/// No token anywhere is the ordinary anonymous case, not an error.
#[test]
fn test_no_token_reads_anonymously() -> Result<(), Box<dyn std::error::Error>> {
    let url = Url::parse("hf://datasets/org/name")?;
    let (config, _path) = url_to_config(&url, env(&[("HF_TOKEN_PATH", "/nonexistent/token")]))?;

    assert!(config.token.is_none());
    Ok(())
}

/// Building a store must succeed for both the anonymous and the credentialed path; a token that
/// cannot become a header value is reported rather than panicking.
#[test]
fn test_make_hf_store() -> Result<(), Box<dyn std::error::Error>> {
    let base = HfConfig {
        repo_id: "org/name".to_string(),
        ..HfConfig::default()
    };

    let _anonymous = super::make_hf_store(base.clone())?;
    let _credentialed = super::make_hf_store(HfConfig {
        token: Some("hf_token".to_string()),
        ..base.clone()
    })?;

    assert!(matches!(
        super::make_hf_store(HfConfig {
            token: Some("bad\nvalue".to_string()),
            ..base
        }),
        Err(HfStoreError::InvalidToken(_))
    ));
    assert!(matches!(
        super::make_hf_store(HfConfig::default()),
        Err(HfStoreError::InvalidUrl(_))
    ));
    Ok(())
}
