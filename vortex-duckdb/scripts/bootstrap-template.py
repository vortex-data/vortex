#!/usr/bin/python3

import os
import re
import shutil
import sys
from pathlib import Path


def is_snake_case(s: str) -> bool:
    """
    Check if the provided string is in snake_case format.
    Snake case is lower case with words separated by underscores, and it can contain digits.

    Args:
        s (str): String to check.

    Returns:
        bool: True if the string is in snake_case, False otherwise.
    """
    pattern = r"^[a-z0-9]+(_[a-z0-9]+)*$"
    return bool(re.match(pattern, s))


def to_camel_case(snake_str: str) -> str:
    """
    Convert a snake_case string to camelCase.

    Args:
        snake_str (str): String in snake_case to convert.

    Returns:
        str: Converted string in camelCase.
    """
    return "".join(x.capitalize() for x in snake_str.lower().split("_"))


def replace(file_name: str, to_find: str, to_replace: str) -> None:
    """
    Replace occurrences of a string within a file, ensuring placeholders are handled.
    The function replaces the `to_find` string with `to_replace`, adds a placeholder,
    and skips lines with placeholders already in place.

    Args:
        file_name (str): Path to the file to perform replacement in.
        to_find (str): String to search for in the file.
        to_replace (str): String to replace `to_find` with.

    Returns:
        None
    """
    with open(file_name, "r", encoding="utf8") as file:
        filedata = file.readlines()

    new_filedata = []
    for line in filedata:
        # Skip lines that have already been replaced by checking for placeholder
        if "__REPLACEMENT_DONE__" in line:
            new_filedata.append(line)
            continue

        modified_line = line.replace(
            to_find,
            to_replace,
        )
        modified_line = modified_line.replace(
            to_find.capitalize(), to_camel_case(to_replace)
        )
        modified_line = modified_line.replace(
            to_find.upper(),
            to_replace.upper(),
        )

        # Add placeholder once after all replacements
        if to_find in line or to_find.capitalize() in line or to_find.upper() in line:
            modified_line += "__REPLACEMENT_DONE__"

        new_filedata.append(modified_line)

    with open(file_name, "w", encoding="utf8") as file:
        file.writelines(new_filedata)


def replace_everywhere(to_find: str, to_replace: str) -> None:
    """
    Replace a string in all files in the project.

    Args:
        to_find (str): String to search for in the file.
        to_replace (str): String to replace `to_find` with.

    Returns:
        None
    """
    for path in files_to_search:
        replace(path, to_find, to_replace)
        replace(path, to_find.capitalize(), to_camel_case(to_replace))
        replace(path, to_find.upper(), to_replace.upper())

    replace("./CMakeLists.txt", to_find, to_replace)
    replace("./Makefile", to_find, to_replace)
    replace("./Makefile", to_find.capitalize(), to_camel_case(to_replace))
    replace("./Makefile", to_find.upper(), to_replace.upper())
    replace("./README.md", to_find, to_replace)
    replace("./extension_config.cmake", to_find, to_replace)
    replace("./scripts/setup-custom-toolchain.sh", to_find, to_replace)
    replace(".github/workflows/MainDistributionPipeline.yml", to_find, to_replace)


def remove_placeholder() -> None:
    """
    Remove the placeholder from all files.

    Returns:
        None
    """
    for path in files_to_search:
        replace_placeholders(path)

    replace_placeholders("./CMakeLists.txt")
    replace_placeholders("./Makefile")
    replace_placeholders("./Makefile")
    replace_placeholders("./Makefile")
    replace_placeholders("./README.md")
    replace_placeholders("./extension_config.cmake")
    replace_placeholders("./scripts/setup-custom-toolchain.sh")


def replace_placeholders(file_name: str) -> None:
    """
    Remove the placeholder from a file.

    Args:
        file_name (str): Path to the file to remove the placeholder from.

    Returns:
        None
    """
    with open(file_name, "r", encoding="utf8") as file:
        filedata = file.read()

    # Remove all placeholders
    filedata = filedata.replace("__REPLACEMENT_DONE__", "")

    with open(file_name, "w", encoding="utf8") as file:
        file.write(filedata)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception(
            "usage: python3 bootstrap-template.py <name_for_extension_in_snake_case>"
        )

    name_extension = sys.argv[1]

    if name_extension[0].isdigit():
        raise Exception("Please dont start your extension name with a number.")

    if not is_snake_case(name_extension):
        raise Exception(
            "Please enter the name of your extension in valid snake_case containing only lower case letters and numbers"
        )

    shutil.copyfile("docs/NEXT_README.md", "README.md")
    os.remove("docs/NEXT_README.md")
    os.remove("docs/README.md")

    files_to_search = []
    files_to_search.extend(Path("./.github").rglob("./**/*.yml"))
    files_to_search.extend(Path("./test").rglob("./**/*.test"))
    files_to_search.extend(Path("./src").rglob("./**/*.hpp"))
    files_to_search.extend(Path("./src").rglob("./**/*.cpp"))
    files_to_search.extend(Path("./src").rglob("./**/*.txt"))
    files_to_search.extend(Path("./src").rglob("./*.md"))

    replace_everywhere("quack", name_extension)
    replace_everywhere("Quack", name_extension.capitalize())
    replace_everywhere("<extension_name>", name_extension)

    remove_placeholder()

    string_to_replace = name_extension
    string_to_find = "quack"

    # rename files
    os.rename(f"test/sql/{string_to_find}.test", f"test/sql/{string_to_replace}.test")
    os.rename(
        f"src/{string_to_find}_extension.cpp", f"src/{string_to_replace}_extension.cpp"
    )
    os.rename(
        f"src/include/{string_to_find}_extension.hpp",
        f"src/include/{string_to_replace}_extension.hpp",
    )

    # remove template-specific files
    os.remove(".github/workflows/ExtensionTemplate.yml")

    # finally, remove this bootstrap file
    os.remove(__file__)
