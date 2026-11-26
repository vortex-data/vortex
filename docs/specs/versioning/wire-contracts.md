# Designing wire contracts

[Backward compatibility](../versioning.md) relies on frozen wire contracts never changing. This page
covers the other direction: what an older reader does with output from a newer writer. A reader that
meets an unknown wire ID fails with an [unknown-ID error](using-editions.md#unknown-ids), which is
safe. The harder case is a newer writer extending a contract under an ID the older reader already
knows. Design each contract so that every extension either decodes correctly on older readers or
fails on them. It must never decode to different values.

These rules apply to array plugins, layouts, and any other serialized component with metadata or
children. Apply them before a format leaves the [unstable edition](editions.md#unstable): after it
is frozen, the contract can no longer change.

## Locating children

The position of every child must follow from the metadata and data type alone:

- **Declare optional children.** Record whether an optional child is present, with a metadata flag
  or an optional message whose presence implies its children. Do not infer presence from the
  number of children. If a third child always meant validity, a later child would be mistaken for
  validity whenever validity is absent.
- **Use a fixed order.** Place required children first and optional children after them, in an
  order that does not depend on which others are present.
- **Count from the start.** Do not locate a child relative to the end of the list, such as
  `children.len() - 1`, because an appended child moves it.

For example, `fastlanes.delta` writes `[bases, deltas]`, followed by its validity child when its
`has_validity` metadata field is set.

A child whose presence is fully determined by the data type needs no flag. For example, the
`vortex.list` layout has a validity child exactly when its data type is nullable.

## Appending children

Readers require at least the children their metadata declares and ignore any that follow. A later
writer may therefore append a child under the same ID only if decoding without it gives the same
values, such as a statistic or an index. A child that changes the decoded values, such as patches
or exceptions, needs a new wire ID, so older readers fail instead of returning wrong values.

Readers do not validate children they ignore.

## Metadata

Protobuf decoding ignores unknown fields, so an older reader drops any field it does not know. A
new field under an existing ID must be safe to ignore. A field that changes how buffers or children
are interpreted needs a new wire ID.

Two consequences follow:

- **Decode metadata as a message, even when it has no fields.** A reader that requires empty
  metadata rejects every later field, including fields that are safe to ignore. A message with no
  fields encodes to zero bytes, so the serialized form is the same. `vortex.union` does this.
- **Make the default value mean the original behavior.** A missing field decodes to its default,
  so a later field's default must describe files written before the field existed.

## Changes that need a new wire ID

Under a frozen ID, any change an older reader would misread or reject needs a new ID:

- adding a child or metadata field that changes the decoded values;
- changing the meaning, order, data type, or length of an existing child or buffer;
- changing the meaning of an existing metadata field;
- writing values that the existing reader rejects.

The writer can keep writing the original ID when the new feature is unused, as BitPacked does with
`fastlanes.bitpacked` and `fastlanes.bitpacked.v2`. An edition then controls when writers may use
the new ID.

## Validating input

A reader must reject malformed input with an error, not a panic, undefined behavior, or an unbounded
allocation:

- Check the data type before calling functions that support only some physical types.
- Derive lengths from the logical length where possible, and require any stored length to match
  exactly. A stored length that is only bounded below lets a small file claim a large physical size.
- Use checked arithmetic for lengths computed from metadata.
- Run these checks before reading children. The in-memory children passed to a deserializer assert
  that each child has the requested data type and length.

Values that need a scan to check, such as offsets that must not decrease, are validated where they
are read.

## Tests

Each contract needs:

- round trips, with and without each optional child, and with sliced input where the format stores
  an offset;
- a test that a trailing extra child is ignored;
- a test for each validation check, which should fail with that check's error;
- a golden file of the metadata bytes, written with `check_metadata`, with every field set to a
  value other than its default.
