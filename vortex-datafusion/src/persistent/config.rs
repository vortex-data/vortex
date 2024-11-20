use chrono::TimeZone as _;
use datafusion::datasource::listing::PartitionedFile;
use object_store::path::Path;
use object_store::ObjectMeta;

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
