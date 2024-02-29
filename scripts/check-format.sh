#!/bin/bash

ALL_FILES=$(
    find . -type f \( -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" -o -name "*.cu" -o -name "*.cuh" \) \
        -not -path "*/external/*"  \
        -not -path "*/build/*"     \
        -not -path "*/.tox/*"      \
    )

ERRORS=0
for file in $ALL_FILES; do
    if ! clang-format --dry-run --Werror "$file"; then
        ((ERRORS = ERRORS + 1))
    fi
done

if [[ $ERRORS == 0 ]]; then
    echo "✅ all files are properly formatted"
else
    echo "❌ there are $ERRORS files not not properly formatted."
    echo "Please run './scripts/format.sh' from the mops root directory"
    exit $ERRORS
fi
