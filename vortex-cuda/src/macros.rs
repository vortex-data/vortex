// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[macro_export]
macro_rules! warn {
    ($($tts:tt)*) => {
        #[cfg(feature = "tracing")]
        {
            tracing::warn!($($tts)*);
        }
    };
}

#[macro_export]
macro_rules! info {
    ($($tts:tt)*) => {
        #[cfg(feature = "tracing")]
        {
            tracing::info!($($tts)*);
        }
    };
}

#[macro_export]
macro_rules! debug {
    ($($tts:tt)*) => {
        #[cfg(feature = "tracing")]
        {
            tracing::info!($($tts)*);
        }
    };
}

#[macro_export]
macro_rules! trace {
    ($($tts:tt)*) => {
        #[cfg(feature = "tracing")]
        {
            tracing::info!($($tts)*);
        }
    };
}
