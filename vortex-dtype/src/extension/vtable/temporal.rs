use vortex_error::{VortexResult, vortex_bail, vortex_err};

use crate::datetime::{TIME_ID, TimeUnit};
use crate::{ExtID, ExtMetadata, ExtensionVTable, vtable};

/// Extension type for the time type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TimeExtensionType {
    /// The time unit for the values.
    pub time_unit: TimeUnit,
}

/// Encoding that captures all of the different temporal types.
///
/// This works with all of the following
///
/// * Time
/// * Date
/// * Timestamp
#[derive(Debug, Copy, Clone)]
pub struct TimeExtensionTypeEncoding;

vtable!(Time);

impl ExtensionVTable for TimeVTable {
    type ExtType = TimeExtensionType;
    type ExtEncoding = TimeExtensionTypeEncoding;

    fn id(extension: &Self::ExtType) -> &ExtID {
        extension.id()
    }

    fn serialize_metadata(extension: &Self::ExtType) -> Option<ExtMetadata> {
        Some(ExtMetadata::new(vec![extension.time_unit as u8].into()))
    }

    fn try_decode(
        id: &ExtID,
        metadata: Option<ExtMetadata>,
    ) -> VortexResult<Option<Self::ExtType>> {
        // bail early if some other extension type
        if id.as_ref() != TIME_ID.as_ref() {
            return Ok(None);
        }

        let Some(meta) = metadata else {
            vortex_bail!("missing metadata for time extension");
        };

        if meta.as_ref().len() != 1 {
            vortex_bail!("TimeExtensionType metadata must be exactly 1 byte");
        }

        let time_unit = TimeUnit::try_from(meta.as_ref()[0])
            .map_err(|err| vortex_err!("invalid TimeUnit in Time metadata: {}", err))?;

        Ok(Some(TimeExtensionType { time_unit }))
    }
}

#[cfg(test)]
mod tests {
    use crate::datetime::{TIME_ID, TimeUnit};
    use crate::extension::vtable::temporal::{TimeExtensionType, TimeVTable};
    use crate::{ExtMetadata, ExtensionVTable, IntoExtensionTypeRef};

    #[test]
    fn test_time_vtable() {
        let time_extension = TimeExtensionType {
            time_unit: TimeUnit::S,
        }
        .into_extension_type_ref();

        assert_eq!(
            time_extension.serialize_metadata().unwrap().as_ref(),
            &[TimeUnit::S as u8]
        );

        let decoded = TimeVTable::try_decode(
            &TIME_ID,
            Some(ExtMetadata::new(vec![TimeUnit::S as u8].into())),
        )
        .unwrap();

        assert_eq!(
            decoded,
            Some(TimeExtensionType {
                time_unit: TimeUnit::S
            })
        );
    }
}
