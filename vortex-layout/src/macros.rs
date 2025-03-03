/// A macro for optionally instrumenting a future, if tracing feature is enabled.
#[macro_export]
macro_rules! instrument {
    ($span_name:expr, $expr:expr) => {
        instrument!($span_name, {}, $expr)
    };
    ($span_name:expr, { $($key:ident = $value:expr),* $(,)? }, $expr:expr) => {
        {
            let task = $expr;
            #[cfg(feature = "tracing")]
            {
                use tracing_futures::Instrument;
                task.instrument(tracing::info_span!(
                    $span_name,
                    $($key = $value,)*
                ))
            }
            #[cfg(not(feature = "tracing"))]
            {
                task
            }
        }
    };
}
