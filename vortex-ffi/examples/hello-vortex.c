#include <stdbool.h>
#include <stdio.h>
#include "vortex.h"

int main(int argc, char *argv[]) {
    // enable logging
    vortex_init_logging(LOG_LEVEL_INFO);

    if (argc < 2) {
        printf("Usage: %s <VORTEX_FILE_URI>\n", argv[0]);
        return 1;
    }

    // Open the file
    char *path = argv[1];
    FileOpenOptions open_opts = {
      .uri = path,
      .property_keys = NULL,
      .property_vals = NULL,
      .property_len = 0,
    };
    printf("Scanning file: %s\n", path);
    File *file = File_open(&open_opts);

    // Start scanning, read new rows.
    ArrayStream *stream = File_scan(file, NULL);
    int chunk = 0;
    while (FFIArrayStream_next(stream)) {
        Array *array = FFIArrayStream_current(stream);
        int len = FFIArray_len(array);
        printf("Chunk %d: %d\n", chunk++, len);
        FFIArray_free(array);
    }

    printf("Scanning complete\n");

    // Cleanup resources.
    FFIArrayStream_free(stream);
    File_free(file);

    return 0;
}
