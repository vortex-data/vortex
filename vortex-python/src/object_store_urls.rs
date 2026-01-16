// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use object_store::ObjectStore;
use object_store::path::Path;
use object_store::registry::DefaultObjectStoreRegistry;
use object_store::registry::ObjectStoreRegistry;
use url::Url;
use vortex::error::VortexResult;

static REGISTRY: LazyLock<DefaultObjectStoreRegistry> =
    LazyLock::new(DefaultObjectStoreRegistry::new);

pub(crate) fn object_store_from_url(url_str: &str) -> VortexResult<(Arc<dyn ObjectStore>, Path)> {
    let parsed_url = Url::parse(url_str)?;
    let (store, path) = REGISTRY.resolve(&parsed_url)?;
    Ok((store, path))
}
