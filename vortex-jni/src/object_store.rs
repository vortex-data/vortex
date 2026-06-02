// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use object_store::ObjectStore;
use url::Url;
use vortex::error::VortexResult;
use vortex::io::compat::Compat;
use vortex::io::filesystem::FileSystemRef;
use vortex::io::object_store::ObjectStoreFileSystem;
use vortex::io::object_store::make_object_store;
use vortex::io::runtime::Handle;
use vortex::utils::aliases::hash_map::HashMap;

pub(crate) fn object_store_fs(
    url: &Url,
    properties: &HashMap<String, String>,
    handle: Handle,
) -> VortexResult<FileSystemRef> {
    let object_store = make_object_store(url, properties)?;
    let object_store = Arc::new(Compat::new(object_store)) as Arc<dyn ObjectStore>;

    Ok(Arc::new(ObjectStoreFileSystem::new(object_store, handle)))
}
