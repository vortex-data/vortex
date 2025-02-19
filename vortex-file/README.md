# Vortex IPC Format

Messages:

* Context - provides configuration context, e.g. which encodings are referenced in the stream.
* Array - indicates the start of an array. Contains the schema.
* Chunk - indices the start of an array chunk. Contains the offsets for each column message.
* ChunkColumn - contains the encoding metadata for a single column of a chunk, including offsets for each buffer.


1. We have a set of sublists, such as
    1. something1
    2. something2
2. We have another set of sublists, such as
    1. something3
    2. something4
