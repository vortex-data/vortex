use datafusion::datasource::listing::PartitionedFile;
use object_store::ObjectMeta;

#[derive(Debug, Clone)]
pub(crate) struct VortexFile {
    pub(crate) object_meta: ObjectMeta,
}

impl From<VortexFile> for PartitionedFile {
    fn from(value: VortexFile) -> Self {
        PartitionedFile::new(value.object_meta.location, value.object_meta.size as u64)
    }
}
