use vortex_error::VortexResult;

use crate::{ExtID, ExtMetadata, ExtensionVTable, vtable};

/// Extension type encoding for all of the temporal types.
#[derive(Debug, Copy, Clone)]
pub struct TemporalExtensionEncoding;

vtable!(Temporal);
//
// pub struct TemporalVTable;

// impl crate::IntoExtensionTypeRef for TemporalVTable {
//     fn into_extension_type_ref(self) -> crate::ExtensionTypeRef {
//         std::sync::Arc::new(unsafe {
//             std::mem::transmute::<TemporalVTable, crate::ExtensionTypeAdapter<TemporalVTable>>(self)
//         })
//     }
// }

pub struct TemporalExtensionType;

impl ExtensionVTable for TemporalVTable {
    type ExtType = TemporalExtensionType;
    type ExtEncoding = ();

    fn id(extension: &Self::ExtType) -> &ExtID {
        todo!()
    }

    fn serialize_metadata(extension: &Self::ExtType) -> Option<ExtMetadata> {
        todo!()
    }

    fn try_decode(
        id: &ExtID,
        metadata: Option<ExtMetadata>,
    ) -> VortexResult<Option<Self::ExtType>> {
        todo!()
    }
}
