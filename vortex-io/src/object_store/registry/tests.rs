// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Write;
use std::sync::Arc;

use object_store::ObjectStore;
use object_store::registry::ObjectStoreRegistry;
use url::Url;

use super::Registry;

/// A registry whose S3 configuration comes from a fixed map rather than the process environment,
/// so these tests neither read nor mutate global state.
fn registry() -> Registry {
    Registry::with_env([("AWS_REGION".to_string(), "us-east-3".to_string())])
}

#[test]
#[expect(clippy::use_debug)]
fn test_resolve_url() -> Result<(), Box<dyn std::error::Error>> {
    let (store, _) = registry().resolve(&Url::parse("s3://my-bucket/test")?)?;

    // NOTE(aduffy): object_store doesn't let us downcast stores, the only way to verify
    //  that a configuration was added was to validate that it ends up in the Debug
    //  output :/
    let mut debug_str = String::new();
    write!(&mut debug_str, "{store:?}")?;

    assert!(debug_str.contains("us-east-3"));
    Ok(())
}

/// Two resolves of the same URL must return the same `Arc` (the cached client) and the same
/// path, and two different keys must get distinct stores. This pins the symmetry the registry
/// promises: cache hit returns the same client and the same key.
#[test]
fn test_resolve_url_caches_per_key() -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry();
    let first = Url::parse("s3://my-bucket/first/second")?;
    let second = Url::parse("s3://my-bucket/first/second")?;

    let (store_a, path_a) = registry.resolve(&first)?;
    let (store_b, path_b) = registry.resolve(&second)?;
    assert!(Arc::ptr_eq(&store_a, &store_b));
    assert_eq!(path_a, path_b);

    // A different bucket gets its own client.
    let other = Url::parse("s3://other-bucket/first/second")?;
    let (store_c, _) = registry.resolve(&other)?;
    assert!(!Arc::ptr_eq(&store_a, &store_c));
    Ok(())
}

/// Two objects in the same bucket must share a cached client. This is the regression that
/// the bespoke "walk every segment then return the whole key" path caused: by walking all
/// segments it stored the store at full depth, so the next resolve saw the whole key
/// already consumed and returned an empty path. Pinning the path here guards against that.
#[test]
fn test_resolve_url_shared_client_same_bucket() -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry();
    let a = Url::parse("s3://my-bucket/path/to/data.vortex")?;
    let b = Url::parse("s3://my-bucket/other/data.vortex")?;

    let (store_a, path_a) = registry.resolve(&a)?;
    let (store_b, path_b) = registry.resolve(&b)?;
    assert!(Arc::ptr_eq(&store_a, &store_b));
    assert_eq!(path_a.as_ref(), "path/to/data.vortex");
    assert_eq!(path_b.as_ref(), "other/data.vortex");
    Ok(())
}

/// Pinning the `entry_at` helper at depth > 0 (the path-prefix case) protects the shared
/// `entry_at` / `cache_and_resolve` helpers from regressing prefix registration.
#[test]
fn test_register_at_prefix_shares_store() -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry();
    let prefix = Url::parse("s3://my-bucket/prefix/")?;
    let (inner, _) = registry.resolve(&prefix)?;
    let cached: Arc<dyn ObjectStore> = Arc::clone(&inner);
    let replaced = registry.register(prefix.clone(), Arc::clone(&cached));
    // First registration at this prefix replaces nothing.
    assert!(replaced.is_none());

    // A resolve at a deeper key still walks through the registered prefix's store.
    let deeper = Url::parse("s3://my-bucket/prefix/inner/key")?;
    let (store_deeper, path_deeper) = registry.resolve(&deeper)?;
    assert!(Arc::ptr_eq(&store_deeper, &cached));
    assert_eq!(path_deeper.as_ref(), "inner/key");
    Ok(())
}

/// A `file://` URL with its path cleared must resolve to a local store. `vortex-duckdb` resolves
/// exactly this shape — it clears the path off the glob's base URL before asking for a
/// filesystem — so this pins the most common local-file case.
#[test]
fn test_resolve_file_url_with_empty_path() -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry();

    let mut base = Url::parse("file:///data/part-0.vortex")?;
    base.set_path("");
    let (store, path) = registry.resolve(&base)?;
    assert!(format!("{store:?}").contains("LocalFileSystem"));
    assert_eq!(path.as_ref(), "");

    // The same store is reused for a second glob rooted at the same authority.
    let (store_again, _) = registry.resolve(&base)?;
    assert!(Arc::ptr_eq(&store, &store_again));
    Ok(())
}

/// An explicitly registered store must win over the build-and-cache fallback, including for
/// schemes `object_store` does not recognize natively. `memory://` stands in for any such scheme
/// so this holds regardless of which service features are enabled.
#[test]
fn test_registered_store_wins_over_build() -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry();
    let registered: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
    registry.register(Url::parse("memory://bucket/")?, Arc::clone(&registered));

    let (store, path) = registry.resolve(&Url::parse("memory://bucket/a/b.vortex")?)?;
    assert!(Arc::ptr_eq(&store, &registered));
    assert_eq!(path.as_ref(), "a/b.vortex");
    Ok(())
}

/// The OpenDAL-backed schemes must resolve through the registry rather than falling through to
/// `parse_url_opts`, which does not recognize them. Without an endpoint the build fails, but the
/// error must come from the OpenDAL builder ("missing required OpenDAL store configuration"), not
/// from `object_store`'s unrecognized-scheme path.
#[cfg(feature = "opendal")]
#[test]
fn test_opendal_schemes_reach_the_opendal_builder() -> Result<(), Box<dyn std::error::Error>> {
    for scheme in vortex_object_store_opendal::SUPPORTED_SCHEMES {
        let url = Url::parse(&format!("{scheme}://bucket/key.vortex"))?;
        let err = registry()
            .resolve(&url)
            .expect_err("no endpoint is configured, so the build must fail");
        let message = err.to_string();
        assert!(
            message.contains("OpenDAL"),
            "{scheme} did not reach the OpenDAL builder: {message}"
        );
    }
    Ok(())
}
