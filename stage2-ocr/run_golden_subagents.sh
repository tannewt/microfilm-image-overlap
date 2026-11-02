#!/usr/bin/env bash
# Resilient subagent transcription driver.
# For each crop: spawn a one-shot `pi -p` subagent to transcribe it.
# - skips jobs whose output already exists and validates (idempotent/resumable)
# - retries failed jobs up to MAX_TRIES times
# - runs up to PARALLEL jobs concurrently
# - never aborts on individual failures; reports a summary at the end

set -u
PARALLEL=3
MAX_TRIES=3
TIMEOUT=900          # per attempt, seconds
MIN_CHARS=60         # minimum valid transcription length
LOG=/tmp/golden_subagents.log
CROPS_ROOT=${1:-/tmp/crops}
OUT_ROOT=${2:-/tmp/golden_chunks}

mkdir -p "$OUT_ROOT"
: > "$LOG"

validate() {
    local f="$1"
    [ -s "$f" ] || return 1
    [ "$(wc -c < "$f")" -ge "$MIN_CHARS" ] || return 1
    # reject obvious markdown/code-fence wrappers
    if head -1 "$f" | grep -q '```'; then return 1; fi
    return 0
}

transcribe_one() {
    local crop="$1"
    local name
    name="$(basename "$crop" .png)"
    local out="$OUT_ROOT/$name.txt"

    if validate "$out"; then
        echo "SKIP  $name (already done)" | tee -a "$LOG"
        return 0
    fi

    for attempt in $(seq 1 "$MAX_TRIES"); do
        echo "RUN   $name attempt $attempt $(date +%H:%M:%S)" >> "$LOG"
        rm -f "$out"
        timeout "$TIMEOUT" pi -p --no-approve "You are a transcription subagent. Read the image file $crop with the read tool. It is a crop from an 1890s American newspaper page photographed at 600 dpi.

Transcribe ALL printed text in it, exactly:
- one output line per printed line of text
- a blank line between paragraphs and between separate articles
- preserve original 1890s spelling, capitalization and punctuation exactly as printed
- a word hyphenated across two printed lines stays split across two lines (write 'com-' on one line, 'plete' on the next)
- include headlines and datelines
- if a word is unreadable, write [?] in its place
- output NOTHING but the transcription text: no commentary, no markdown fences, no headers

Use the write tool to save the transcription to $out , then stop. Do not explain anything." \
            > /dev/null 2>> "$LOG"
        rc=$?
        if [ $rc -eq 0 ] && validate "$out"; then
            echo "OK    $name (attempt $attempt, $(wc -w < "$out") words)" | tee -a "$LOG"
            return 0
        fi
        echo "FAIL  $name attempt $attempt rc=$rc" >> "$LOG"
        sleep 5
    done
    echo "DEAD  $name (gave up after $MAX_TRIES attempts)" | tee -a "$LOG"
    return 1
}

# collect jobs
col_selected() {
    case "$1" in
        06818-*_c[01]_r*) return 0 ;;   # P81-4720: columns 0,1
        04860-*_c[12]_r*) return 0 ;;   # P81-4721: columns 1,2
        07278-*_c[12]_r*) return 0 ;;   # P81-4722: columns 1,2
        *) return 1 ;;
    esac
}

jobs_list=()
for page_dir in "$CROPS_ROOT"/*/; do
    for crop in "$page_dir"/*_c*_r*.png; do
        [ -e "$crop" ] || continue
        if col_selected "$(basename "$crop")"; then
            jobs_list+=("$crop")
        fi
    done
done

echo "Dispatching ${#jobs_list[@]} transcription jobs..."

running=0
failed=0
pids=()
for crop in "${jobs_list[@]}"; do
    transcribe_one "$crop" &
    pids+=($!)
    running=$((running + 1))
    if [ "$running" -ge "$PARALLEL" ]; then
        for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
        pids=(); running=0
    fi
done
for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p" || failed=$((failed + 1)); done

ok=0
for crop in "${jobs_list[@]}"; do
    name="$(basename "$crop" .png)"
    if validate "$OUT_ROOT/$name.txt"; then ok=$((ok + 1)); fi
done

echo
echo "=== Summary: $ok/${#jobs_list[@]} crops transcribed, $failed job groups failed ==="
exit 0