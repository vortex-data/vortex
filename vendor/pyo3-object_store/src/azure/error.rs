const STORE: &str = "MicrosoftAzure";

// Vendored from upstream
/// A specialized `Error` for Azure builder-related errors
#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("Failed parsing an SAS key")]
    DecodeSasKey { source: std::str::Utf8Error },

    #[error("Missing component in SAS query pair")]
    MissingSasComponent {},
}

impl From<Error> for object_store::Error {
    fn from(source: Error) -> Self {
        Self::Generic {
            store: STORE,
            source: Box::new(source),
        }
    }
}
