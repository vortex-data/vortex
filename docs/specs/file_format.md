# File Format

:::{seealso}
It is recommended that you familiarize yourself with [Vortex Layouts](/concepts/layouts) prior to reading this document.
:::

Recall that [Vortex Layouts](/concepts/layouts) provide a mechanism to efficiently query large serialized Vortex
arrays. The _Vortex File Format_ is designed to provide a container for these serialized arrays, as well as footer
definition that allows efficiently querying the layout.

Other considerations for the Vortex file format include:

* Backwards compatibility, and (uniquely) forwards compatibility.
* Fine-grained encryption.
* Efficient access for both local disk and cloud storage.
* Minimal overhead reading few columns or rows from wide or long arrays.

## File Specification

The Vortex file format has a very small definition, with much of the complexity encapsulated
in [Vortex Layouts](/concepts/layouts).

```plaintext
0..4: 'VTXF' magic number
... segments of binary data, optionally with inter-segment padding
... postfix data
-8..-6: u16 version tag
-6..-4: u16 postfix length
-4..: 'VTXF' magic number
```

The file format begins and ends with the 4-byte magic number `VTXF`.
Immediately prior to the trailing magic number are two 16-bit integers: the version tag and the length of the postfix.

### Postfix

The postfix contains

## Footer Specification

The footer

With this in mind, the idea of a Vortex file format becomes simply a way to load a `LayoutData` tree whose
segments are stored within the file.

The Vortex file format provides a way to store such a layout in a contiguous binary file and optimize access
from memory-mapped, local disk, or cloud storage.

is a way to store these serialized arrays in a contiguous binary format.

a way to represent an array that can lazily load portions of data.

The goal of the Vortex file format is to provide a backwards- and forwards-compatible way to store and query large
Vortex arrays serialized in a contiguous binary format.

