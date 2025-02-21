//! Persistent implementation of a Vortex table provider.
mod cache;
mod config;
mod execution;
mod format;
mod metrics;
mod opener;
mod sink;

pub use format::{VortexFormat, VortexFormatFactory, VortexFormatOptions};

#[cfg(test)]
/// Utility function to register Vortex with a [`SessionStateBuilder`]
fn register_vortex_format_factory(
    factory: VortexFormatFactory,
    session_state_builder: &mut datafusion::execution::SessionStateBuilder,
) {
    if let Some(table_factories) = session_state_builder.table_factories() {
        table_factories.insert(
            datafusion_common::GetExt::get_ext(&factory).to_uppercase(), // Has to be uppercase
            std::sync::Arc::new(datafusion::datasource::provider::DefaultTableFactory::new()),
        );
    }

    if let Some(file_formats) = session_state_builder.file_formats() {
        file_formats.push(std::sync::Arc::new(factory));
    }
}
