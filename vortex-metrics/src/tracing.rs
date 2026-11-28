// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Global tracing and instrumentation

use parking_lot::RwLock;

static LABELS: RwLock<Vec<(&str, String)>> = RwLock::new(Vec::new());

/// Set global labels used for external tracing.
pub fn set_global_labels<I>(i: I)
where
    I: IntoIterator<Item = (&'static str, String)>,
{
    let new = Vec::from_iter(i);
    *LABELS.write() = new;
}

/// Get the globally set labels for external tracing.
pub fn get_global_labels() -> Vec<(&'static str, String)> {
    LABELS.read().clone()
}

#[cfg(test)]
mod tests {
    use super::get_global_labels;
    use super::set_global_labels;

    #[test]
    fn global_labels_set_get_overwrite_and_clear() {
        set_global_labels([("env", "test".to_owned()), ("service", "vortex".to_owned())]);
        assert_eq!(
            get_global_labels(),
            vec![("env", "test".to_owned()), ("service", "vortex".to_owned())]
        );

        set_global_labels([("env", "prod".to_owned())]);
        assert_eq!(get_global_labels(), vec![("env", "prod".to_owned())]);

        set_global_labels(std::iter::empty());
        assert!(get_global_labels().is_empty());
    }
}
