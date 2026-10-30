#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Collect metrics and generate HTML report after all cloud workers finish.
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

KIND=""
INPUT_ROOT=""
OUTPUT_DIR=""
MODE=""
GROUP_BY=()
TTEST="none"
BASELINE_QUERY=""
PAIR_BY=()
TTEST_GROUP_BY=()
TITLE=""
PORT="8080"
SERVE=0
DRY_RUN=0

print_usage() {
    cat <<'USAGE'
Usage:
  bash scripts/experiments/cloud/collect_reports.sh \
      --kind ofat \
      --input_root /workspace/ogd_experiments/ofat_full

  bash scripts/experiments/cloud/collect_reports.sh \
      --kind cv \
      --input_root /workspace/ogd_experiments/cv4_full

Options:
  --kind ofat|cv|generic
  --input_root PATH
  --output_dir PATH
      Default: <input_root>/reports
  --serve
  --port N
  --dry_run

Advanced:
  --ttest none|paired|independent
  --baseline_query key=value,...
  --group_by ...
  --pair_by ...
  --ttest_group_by ...
USAGE
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

require_value() {
    local key="$1"
    local value="${2:-}"
    [[ -n "$value" ]] || fail "$key requires a value."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kind)
            require_value "$1" "${2:-}"
            KIND="$2"
            shift 2
            ;;
        --input_root)
            require_value "$1" "${2:-}"
            INPUT_ROOT="$2"
            shift 2
            ;;
        --output_dir)
            require_value "$1" "${2:-}"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --mode)
            require_value "$1" "${2:-}"
            MODE="$2"
            shift 2
            ;;
        --group_by)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                GROUP_BY+=("$1")
                shift
            done
            ;;
        --ttest)
            require_value "$1" "${2:-}"
            TTEST="$2"
            shift 2
            ;;
        --baseline_query)
            require_value "$1" "${2:-}"
            BASELINE_QUERY="$2"
            shift 2
            ;;
        --pair_by)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                PAIR_BY+=("$1")
                shift
            done
            ;;
        --ttest_group_by)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                TTEST_GROUP_BY+=("$1")
                shift
            done
            ;;
        --title)
            require_value "$1" "${2:-}"
            TITLE="$2"
            shift 2
            ;;
        --serve)
            SERVE=1
            shift
            ;;
        --port)
            require_value "$1" "${2:-}"
            PORT="$2"
            shift 2
            ;;
        --dry_run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            fail "Unknown argument: $1"
            ;;
    esac
done

[[ -n "$KIND" ]] || fail "--kind is required."
[[ -n "$INPUT_ROOT" ]] || fail "--input_root is required."

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="${INPUT_ROOT%/}/reports"
fi

case "$KIND" in
    ofat)
        MODE="${MODE:-sweep}"
        if [[ "${#GROUP_BY[@]}" -eq 0 ]]; then
            GROUP_BY=(param value)
        fi
        TITLE="${TITLE:-OGD-LBS OFAT Sweep Report}"
        ;;
    cv)
        MODE="${MODE:-cv}"
        if [[ "${#GROUP_BY[@]}" -eq 0 ]]; then
            GROUP_BY=(split)
        fi
        if [[ "$TTEST" == "none" ]]; then
            TTEST="paired"
            BASELINE_QUERY="${BASELINE_QUERY:-split=train}"
            if [[ "${#PAIR_BY[@]}" -eq 0 ]]; then
                PAIR_BY=(model fold_index)
            fi
            if [[ "${#TTEST_GROUP_BY[@]}" -eq 0 ]]; then
                TTEST_GROUP_BY=(split)
            fi
        fi
        TITLE="${TITLE:-OGD-LBS 4-Fold CV Report}"
        ;;
    generic)
        MODE="${MODE:-generic}"
        TITLE="${TITLE:-OGD-LBS Experiment Report}"
        ;;
    *)
        fail "--kind must be ofat, cv, or generic"
        ;;
esac

AGG_CMD=(
    python scripts/experiments/aggregate_metrics.py
    --mode "$MODE"
    --input_root "$INPUT_ROOT"
    --output_dir "$OUTPUT_DIR"
)

if [[ "${#GROUP_BY[@]}" -gt 0 ]]; then
    AGG_CMD+=(--group_by "${GROUP_BY[@]}")
fi

if [[ "$TTEST" != "none" ]]; then
    AGG_CMD+=(--ttest "$TTEST")
    [[ -n "$BASELINE_QUERY" ]] && AGG_CMD+=(--baseline_query "$BASELINE_QUERY")
    [[ "${#PAIR_BY[@]}" -gt 0 ]] && AGG_CMD+=(--pair_by "${PAIR_BY[@]}")
    [[ "${#TTEST_GROUP_BY[@]}" -gt 0 ]] && AGG_CMD+=(--ttest_group_by "${TTEST_GROUP_BY[@]}")
fi

REPORT_HTML="$OUTPUT_DIR/OGD_LBS_Report.html"

REPORT_CMD=(
    python scripts/experiments/generate_report.py
    --aggregate_dir "$OUTPUT_DIR"
    --mode "$MODE"
    --output_html "$REPORT_HTML"
    --title "$TITLE"
)

if [[ "$SERVE" -eq 1 ]]; then
    REPORT_CMD+=(--serve --port "$PORT")
fi

echo "================================================================================"
echo "OGD-LBS collect reports"
echo "PROJECT_ROOT : $PROJECT_ROOT"
echo "KIND         : $KIND"
echo "MODE         : $MODE"
echo "INPUT_ROOT   : $INPUT_ROOT"
echo "OUTPUT_DIR   : $OUTPUT_DIR"
echo "TITLE        : $TITLE"
echo "TTEST        : $TTEST"
echo "DRY_RUN      : $DRY_RUN"
echo "================================================================================"

echo
echo "[aggregate command]"
printf ' %q' "${AGG_CMD[@]}"
echo

echo
echo "[report command]"
printf ' %q' "${REPORT_CMD[@]}"
echo

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo
    echo "[dry-run] skipped aggregation/report generation"
    exit 0
fi

"${AGG_CMD[@]}"
"${REPORT_CMD[@]}"

echo
echo "================================================================================"
echo "Report generated:"
echo "  $REPORT_HTML"
echo "================================================================================"
