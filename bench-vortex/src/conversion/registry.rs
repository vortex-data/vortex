// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Registry for format converters

use std::sync::Arc;

use crate::Format;
use crate::conversion::converters::ParquetToVortexConverter;
use crate::conversion::{FormatConverter, converters};

/// Registry of available format converters
///
/// This provides a central place to look up converters for specific
/// format pairs, avoiding the need for each benchmark to know about
/// all possible converters.
pub struct ConverterRegistry {
    converters: Vec<Arc<dyn FormatConverter>>,
}

impl ConverterRegistry {
    /// Create a new registry with all built-in converters
    pub fn new() -> Self {
        let mut registry = Self {
            converters: Vec::new(),
        };

        // Register Parquet to Vortex converters
        registry.register(Arc::new(ParquetToVortexConverter::new(
            Format::OnDiskVortex,
        )));
        registry.register(Arc::new(ParquetToVortexConverter::new(
            Format::VortexCompact,
        )));

        // Register Lance converter if feature is enabled
        #[cfg(feature = "lance")]
        registry.register(Arc::new(converters::ParquetToLanceConverter::new()));

        // Register identity converters for formats that don't need conversion
        registry.register(Arc::new(converters::IdentityConverter::new(
            Format::Parquet,
        )));
        registry.register(Arc::new(converters::IdentityConverter::new(
            Format::OnDiskDuckDB,
        )));

        registry
    }

    /// Register a converter
    pub fn register(&mut self, converter: Arc<dyn FormatConverter>) {
        self.converters.push(converter);
    }

    /// Find a converter for the given format pair
    pub fn find_converter(
        &self,
        source_format: Format,
        target_format: Format,
    ) -> Option<Arc<dyn FormatConverter>> {
        self.converters
            .iter()
            .find(|c| c.supports(source_format, target_format))
            .cloned()
    }

    /// List all available converters
    pub fn list_converters(&self) -> Vec<String> {
        self.converters
            .iter()
            .map(|c| {
                format!(
                    "{}: {} -> {}",
                    c.name(),
                    c.source_format(),
                    c.target_format()
                )
            })
            .collect()
    }

    /// Check if a conversion is supported
    pub fn supports_conversion(&self, source_format: Format, target_format: Format) -> bool {
        self.find_converter(source_format, target_format).is_some()
    }
}

impl Default for ConverterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global converter registry instance
pub fn global_registry() -> &'static ConverterRegistry {
    static REGISTRY: std::sync::OnceLock<ConverterRegistry> = std::sync::OnceLock::new();
    REGISTRY.get_or_init(ConverterRegistry::new)
}
