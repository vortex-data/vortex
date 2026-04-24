# Vortex Java bindings

We provide two interfaces for working with Vortex from Java:

- `vortex-java` - a low-level interface JNI for working with Vortex files and arrays on cloud and local storage
- `vortex-spark` - A Spark connector for working with datasets of Vortex files

## Publishing

We publish three artifacts out of this repo at CI time to Maven Central Sonatype:

* `vortex-jni` JAR containing the JNI code, plus compiled native libraries for all of the following targets: `aarch64-apple-darwin`, `aarch64-unknown-linux-gnu`, `x86_64-unknown-linux-gnu`
* `vortex-jni-all` which is the "shadow JAR" containing all of `vortex-jni` as well as all upstream Java dependencies packaged in a single JAR.
* `vortex-spark` which is the runtime JAR needed for the Vortex Spark bindings

We use the [following GPG key](https://keyserver.ubuntu.com/pks/lookup?search=8745D1A87C0B2159&fingerprint=on&op=index) for publishing:

```
-----BEGIN PGP PUBLIC KEY BLOCK-----

mDMEZ/0bQRYJKwYBBAHaRw8BAQdACooQXQF2hAgoEdDtFaMaFMuVVwC/EE5WNKUr
80aZmvK0P1ZvcnRleCAoVm9ydGV4IE1hdmVuIENlbnRyYWwgcHVibGlzaCBrZXkp
IDx2b3J0ZXhAc3BpcmFsZGIuY29tPoiTBBMWCgA7FiEEAqFvWj3Y1EBr8ebQh0XR
qHwLIVkFAmf9G0ECGwMFCwkIBwICIgIGFQoJCAsCBBYCAwECHgcCF4AACgkQh0XR
qHwLIVmALgD6A9yZ/s9v/TQxmw3Pp8FlKHUMenQWqJefNUz9VHhSoA4A/0i+dqYx
r+LSBcohX00O/CHGhzr5CaxNH7SVdaP4XjkBuDgEZ/0bQRIKKwYBBAGXVQEFAQEH
QL3jcgwAKKq3MQR+YGhCC+od0dDVtqt3u1sFNuS98KBFAwEIB4h4BBgWCgAgFiEE
AqFvWj3Y1EBr8ebQh0XRqHwLIVkFAmf9G0ECGwwACgkQh0XRqHwLIVnt5wEAn54g
t062oWUzNLPcHRzOTjDVzAiUzj5wqJvbiSjttCUBAN4jEfOJPyGKwUcK8zurQTT+
vWBCujQBRqlcCGIIawcI
=qWSF
-----END PGP PUBLIC KEY BLOCK-----
```

The private key and passphrase for the publish key are owned by the Vortex Dev Team.

