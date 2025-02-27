plugins {
    id("org.gradle.toolchains.foojay-resolver") version "0.9.0"
}

toolchainManagement {
    jvm {
        javaRepositories {
            repository("amazon-corretto") {
                resolverClass.set(org.gradle.toolchains.foojay.FoojayToolchainResolver::class.java)
            }
        }
    }
}

rootProject.name = "vortex-root"

// API bindings
include("vortex-jni")
include("vortex-spark")

// Integration tests
// include("ete")
