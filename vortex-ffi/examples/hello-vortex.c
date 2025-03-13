#include <stdbool.h>
#include <stdio.h>

/* Type declarations */
typedef struct File* File;
typedef struct ArrayStream* ArrayStream;
typedef struct Array* Array;

/* Method declarations */
extern File File_open(const char* path);
extern ArrayStream File_scan(File file, const void* options);
extern void File_free(File file);
extern bool FFIArrayStream_next(ArrayStream stream);
extern Array FFIArrayStream_current(ArrayStream stream);
extern void FFIArrayStream_free(ArrayStream stream);
extern int FFIArray_len(Array array);
extern void FFIArray_free(Array array);

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <VORTEX_FILE>\n", argv[0]);
        return 1;
    }

    // Open the file
    char *path = argv[1];
    printf("Scanning file: %s\n", path);
    File file = File_open(path);

    // Start scanning, read new rows.
    ArrayStream stream = File_scan(file, NULL);
    int chunk = 0;
    while (FFIArrayStream_next(stream)) {
        Array array = FFIArrayStream_current(stream);
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
