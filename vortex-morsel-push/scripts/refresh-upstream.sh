#!/usr/bin/env bash

set -euo pipefail

EXACT_IMPORT_SOURCE_REF=ae8b9800409a60d1ceebb2b8181a144581a0cc45
EXACT_IMPORT_COMMIT=e592bf4269add47dfb3994d105e55652cae30503
CURRENT_UPSTREAM_REF=$EXACT_IMPORT_SOURCE_REF
UPSTREAM_SUBTREE=vortex-morsel
LOCAL_SUBTREE=vortex-morsel-push

usage() {
    echo "usage: $0 [--check NEW_SOURCE_REF | --apply NEW_SOURCE_REF]" >&2
}

die() {
    echo "refresh-upstream: $*" >&2
    exit 1
}

is_allowed_overlay() {
    case "$1" in
        Cargo.toml | README.md | UPSTREAM.md | scripts/refresh-upstream.sh | \
            src/bin/morsel-eval.rs | \
            src/bin/tpch-eval.rs | src/build.rs | src/driver.rs | src/executor.rs | \
            src/lib.rs | src/nodes/conjunct.rs)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

mode=verify
new_source_ref=
case "$#" in
    0) ;;
    2)
        mode=$1
        new_source_ref=$2
        case "$mode" in
            --check | --apply) ;;
            *)
                usage
                exit 2
                ;;
        esac
        ;;
    *)
        usage
        exit 2
        ;;
esac

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || die "not inside a Git worktree"
cd "$repo_root"

if [[ -n $(git status --porcelain=v1 --untracked-files=all -- "$LOCAL_SUBTREE") ]]; then
    die "$LOCAL_SUBTREE must be clean before verification or refresh"
fi

git cat-file -e "$EXACT_IMPORT_SOURCE_REF:$UPSTREAM_SUBTREE" 2>/dev/null ||
    die "recorded import source $EXACT_IMPORT_SOURCE_REF:$UPSTREAM_SUBTREE is unavailable"
git cat-file -e "$EXACT_IMPORT_COMMIT:$LOCAL_SUBTREE" 2>/dev/null ||
    die "recorded import tree $EXACT_IMPORT_COMMIT:$LOCAL_SUBTREE is unavailable"
git cat-file -e "$CURRENT_UPSTREAM_REF:$UPSTREAM_SUBTREE" 2>/dev/null ||
    die "current source tree $CURRENT_UPSTREAM_REF:$UPSTREAM_SUBTREE is unavailable"

if ! git diff --quiet \
    "$EXACT_IMPORT_SOURCE_REF:$UPSTREAM_SUBTREE" \
    "$EXACT_IMPORT_COMMIT:$LOCAL_SUBTREE"; then
    die "recorded import commit is not an exact copy of the recorded upstream tree"
fi

unexpected_overlay=()
while IFS= read -r path; do
    if [[ -n $path ]] && ! is_allowed_overlay "$path"; then
        unexpected_overlay+=("$path")
    fi
done < <(
    git diff --name-only \
        "$CURRENT_UPSTREAM_REF:$UPSTREAM_SUBTREE" \
        "HEAD:$LOCAL_SUBTREE"
)

if ((${#unexpected_overlay[@]} != 0)); then
    printf 'refresh-upstream: unexpected integration overlay path: %s\n' \
        "${unexpected_overlay[@]}" >&2
    exit 1
fi

echo "Verified exact push import and integration overlay allowlist."

if [[ $mode == verify ]]; then
    exit 0
fi

git rev-parse --verify --quiet "${new_source_ref}^{commit}" >/dev/null ||
    die "new source ref $new_source_ref is not a local commit"
git cat-file -e "$new_source_ref:$UPSTREAM_SUBTREE" 2>/dev/null ||
    die "new source tree $new_source_ref:$UPSTREAM_SUBTREE is unavailable"

patch_file=$(mktemp "${TMPDIR:-/tmp}/vortex-morsel-push-refresh.XXXXXX.patch")
trap 'rm -f "$patch_file"' EXIT
git diff --binary \
    "$CURRENT_UPSTREAM_REF:$UPSTREAM_SUBTREE" \
    "$new_source_ref:$UPSTREAM_SUBTREE" >"$patch_file"

if [[ ! -s $patch_file ]]; then
    echo "No upstream changes between $CURRENT_UPSTREAM_REF and $new_source_ref."
    exit 0
fi

git apply --check --3way --directory="$LOCAL_SUBTREE" "$patch_file"
if [[ $mode == --check ]]; then
    echo "Push refresh from $new_source_ref applies cleanly (dry run only)."
    exit 0
fi

git apply --3way --directory="$LOCAL_SUBTREE" "$patch_file"
echo "Applied push refresh from $new_source_ref."
echo "Review the overlay, then update CURRENT_UPSTREAM_REF and UPSTREAM.md."
