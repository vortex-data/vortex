// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

fn main() {
    println!("cargo:rerun-if-changed=cpp/fsst8_wrapper.cpp");
    println!("cargo:rerun-if-changed=cpp/fsst12_wrapper.cpp");
    println!("cargo:rerun-if-changed=cpp/onpair_cpp_wrapper.cpp");
    println!("cargo:rerun-if-changed=vendor/fsst_cpp");
    println!("cargo:rerun-if-changed=vendor/onpair_cpp");

    if std::env::var("CARGO_FEATURE_FSST_CPP").is_ok() {
        let mut build = cc::Build::new();
        build
            .cpp(true)
            .std("c++17")
            .file("cpp/fsst8_wrapper.cpp")
            .file("cpp/fsst12_wrapper.cpp")
            .include("vendor/fsst_cpp")
            .define("NONOPT_FSST", "1")
            // FSST-12 has a portability nit (`unsigned long*` vs
            // `unsigned long long*` arg types) that is only a warning under
            // -fpermissive; the underlying sizes are identical on every
            // platform we run on.
            .flag_if_supported("-fpermissive")
            .flag_if_supported("-Wno-everything")
            .warnings(false)
            .extra_warnings(false);
        build.compile("fsst_cpp");
    }

    if std::env::var("CARGO_FEATURE_ONPAIR_CPP").is_ok() {
        let mut build = cc::Build::new();
        build
            .cpp(true)
            .std("c++20")
            .file("cpp/onpair_cpp_wrapper.cpp")
            .file("vendor/onpair_cpp/src/onpair/column/column.cpp")
            .file("vendor/onpair_cpp/src/onpair/core/dictionary_view.cpp")
            .file("vendor/onpair_cpp/src/onpair/encoding/parsing/parser.cpp")
            .file("vendor/onpair_cpp/src/onpair/encoding/training/trainer.cpp")
            .include("vendor/onpair_cpp/include")
            // Upstream uses a non-portable Boost flat_map; require Boost on
            // the system include path. The Debian/Ubuntu package is
            // `libboost-dev`; most build environments already have it.
            .warnings(false)
            .extra_warnings(false);
        build.compile("onpair_cpp");
    }
}
