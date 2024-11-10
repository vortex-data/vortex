use std::sync::Arc;

use arrow_schema::SchemaRef;
use chrono::TimeZone as _;
use datafusion::datasource::listing::PartitionedFile;
use object_store::path::Path;
use object_store::ObjectMeta;
use vortex_array::Context;

#[derive(Debug, Clone)]
pub struct VortexFile {
    pub(crate) object_meta: ObjectMeta,
}

impl From<VortexFile> for PartitionedFile {
    fn from(value: VortexFile) -> Self {
        PartitionedFile::new(value.object_meta.location, value.object_meta.size as u64)
    }
}

impl VortexFile {
    pub fn new(path: impl Into<String>, size: u64) -> Self {
        Self {
            object_meta: ObjectMeta {
                location: Path::from(path.into()),
                last_modified: chrono::Utc.timestamp_nanos(0),
                size: size as usize,
                e_tag: None,
                version: None,
            },
        }
    }
}

#[derive(Debug, Default)]
pub struct VortexTableOptions {
    pub(crate) data_files: Vec<VortexFile>,
    pub(crate) schema: Option<SchemaRef>,
    pub(crate) ctx: Arc<Context>,
}

impl VortexTableOptions {
    pub fn new(schema: SchemaRef, data_files: Vec<VortexFile>, ctx: Arc<Context>) -> Self {
        Self {
            data_files,
            schema: Some(schema),
            ctx,
        }
    }
}
