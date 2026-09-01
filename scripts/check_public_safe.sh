#!/usr/bin/env bash
# check_public_safe.sh -- public-safety gate for a repo that will be made public.
#
# Fails, with file:line output, when a scanned file carries a credential
# marker, an absolute home path, or a maintainer-seeded identity string.
#
#   check_public_safe.sh          scan the staged files (the pre-commit default)
#   check_public_safe.sh --all    scan every tracked file (CI / audit)
#
# Two pattern sets:
#
#   built-in   Generic and repo-independent: credential prefixes, an address
#              marker, absolute home prefixes. Assembled at run time from
#              fragments so this script never contains, literally, the path
#              prefixes it forbids. The credential prefixes are inert on their
#              own but still match, so this script exempts its own path from
#              the built-in scan -- and only from that scan.
#
#   extended   Identity strings (a name, the private sibling repo's name, an
#              email local-part). Never built in and never tracked. Copy
#              .public-safety-patterns.example to .public-safety-patterns
#              (gitignored) and put the real values there, one extended regex
#              per line.
#
# Documentation of the gate is not a leak: `docs/research/` records quote the
# patterns they were built from. Those lines are listed, one per line with a
# reason, in the tracked .public-safety-allow. An allow entry is `path:line`.
# In --all mode an entry that no longer matches anything is reported as stale
# and fails the run, so the list cannot silently drift into a blanket pass.
#
# Matched text is redacted to its first four characters in the output, so a
# finding never echoes a live secret into a terminal or a CI log. The file:line
# is exact; open the file to see the rest.
#
# Exit 0 clean, 1 on a finding or a stale allow entry, 2 on a usage error.

set -euo pipefail

self_path='scripts/check_public_safe.sh'
allow_file='.public-safety-allow'
extra_file='.public-safety-patterns'

usage() {
    printf 'usage: %s [--all]\n' "${0##*/}" >&2
}

mode=staged
case "${1-}" in
    '') ;;
    --all) mode=all ;;
    -h | --help)
        usage
        exit 0
        ;;
    *)
        usage
        exit 2
        ;;
esac

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

# Built from fragments: the literal prefixes must not appear in a tracked file.
mac_home="$(printf '/Use%s' 'rs/')"
nix_home="$(printf '/ho%s' 'me/')"
mail_marker="$(printf '@gm%s' 'ail')"

builtin_patterns=(
    'ghp_'
    'gho_'
    'ghs_'
    'github_pat_'
    'AKIA'
    'ASIA'
    '-----BEGIN'
    'xoxb-'
    'sk-[A-Za-z0-9]{20}'
    "$mail_marker"
    "$mac_home"
    "$nix_home"
)

join_alt() {
    local IFS='|'
    printf '%s' "$*"
}

builtin_re="$(join_alt "${builtin_patterns[@]}")"

# --- extended patterns -------------------------------------------------------
extra_patterns=()
if [ -f "$extra_file" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%$'\r'}"
        case "$line" in
            '' | '#'*) continue ;;
        esac
        extra_patterns+=("$line")
    done < "$extra_file"
fi

# --- allow list --------------------------------------------------------------
allow_keys=''
if [ -f "$allow_file" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%$'\r'}"
        case "$line" in
            '' | '#'*) continue ;;
        esac
        allow_keys="$allow_keys
$line"
    done < "$allow_file"
fi

in_list() { # $1 = needle, $2 = newline-separated haystack
    case "
$2
" in
        *"
$1
"*) return 0 ;;
    esac
    return 1
}

# --- file list ---------------------------------------------------------------
files=()
while IFS= read -r -d '' f; do
    [ -f "$f" ] || continue
    files+=("$f")
done < <(
    if [ "$mode" = all ]; then
        git ls-files -z
    else
        git diff --cached --name-only -z --diff-filter=ACMR
    fi
)

builtin_files=()
for f in ${files[@]+"${files[@]}"}; do
    [ "$f" = "$self_path" ] || builtin_files+=("$f")
done

scan() { # $1 = extended regex, rest = files -> "path:line:match" per match
    local re="$1"
    shift
    [ "$#" -gt 0 ] || return 0
    grep -onIHE -e "$re" -- "$@" || true
}

raw="$(
    scan "$builtin_re" ${builtin_files[@]+"${builtin_files[@]}"}
    for p in ${extra_patterns[@]+"${extra_patterns[@]}"}; do
        scan "$p" ${files[@]+"${files[@]}"}
    done
)"

redact() {
    local m="$1"
    if [ "${#m}" -gt 4 ]; then
        printf '%s...' "${m:0:4}"
    else
        printf '%s' "$m"
    fi
}

findings=''
seen_allow=''
while IFS= read -r hit; do
    [ -n "$hit" ] || continue
    path="${hit%%:*}"
    rest="${hit#*:}"
    lineno="${rest%%:*}"
    match="${rest#*:}"
    key="$path:$lineno"
    if in_list "$key" "$allow_keys"; then
        in_list "$key" "$seen_allow" || seen_allow="$seen_allow
$key"
        continue
    fi
    findings="$findings
$key: $(redact "$match")"
done <<< "$raw"

stale=''
if [ "$mode" = all ]; then
    while IFS= read -r key; do
        [ -n "$key" ] || continue
        in_list "$key" "$seen_allow" || stale="$stale
$key"
    done <<< "$allow_keys"
fi

status=0

if [ -n "$findings" ]; then
    printf 'public-safety: forbidden content (match redacted to 4 chars)\n' >&2
    printf '%s\n' "$findings" | sed '/^$/d' | sort -u >&2
    status=1
fi

if [ -n "$stale" ]; then
    printf 'public-safety: stale %s entries -- these no longer match, re-verify and remove\n' "$allow_file" >&2
    printf '%s\n' "$stale" | sed '/^$/d' | sort -u >&2
    status=1
fi

exit "$status"
