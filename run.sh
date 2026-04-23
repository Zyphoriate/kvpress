#!/usr/bin/env bash

set -euo pipefail

on_interrupt() {
	echo
	echo "[$(date '+%F %T')] Interrupted by Ctrl+C, exiting immediately."
	exit 130
}

trap on_interrupt INT TERM

# Usage:
#   ./run.sh <dataset/data_dir> <press_name> <gpu_ids_csv> [model]
# Example:
#   ./run.sh infinitebench/passkey chunckkv 2,3,4,5

if [[ $# -lt 3 ]]; then
	echo "Usage: $0 <dataset/data_dir> <press_name> <gpu_ids_csv> [model]"
	exit 1
fi

target="$1"
press_name="$2"
gpu_ids="$3"
model="${4:-Qwen/Qwen3-8B}"

if [[ "$target" == */* ]]; then
	dataset="${target%%/*}"
	data_dir="${target#*/}"
else
	dataset="$target"
	data_dir=""
fi

# Common typo compatibility.
if [[ "$press_name" == "chunckkv" ]]; then
	press_name="chunkkv"
fi

# Full ratio sweep used in this workspace.
compression_ratios=(0.1 0.3 0.5 0.7 0.9)

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eval_script="$script_dir/evaluation/evaluate.py"
output_root="$script_dir/results"

has_existing_results() {
	local dataset="$1"
	local data_dir="$2"
	local model="$3"
	local press_name="$4"
	local ratio="$5"

	local ratio_fmt
	ratio_fmt=$(printf "%.2f" "$ratio")
	local model_tag="${model//\//--}"
	local result_dir_name

	if [[ -n "$data_dir" ]]; then
		result_dir_name="${dataset}__${data_dir}__${model_tag}__${press_name}__${ratio_fmt}"
	else
		result_dir_name="${dataset}__${model_tag}__${press_name}__${ratio_fmt}"
	fi

	local base_dir="$output_root/$result_dir_name"

	# Remove empty subdirectories before checking (leave files untouched).
	if [[ -d "$output_root" ]]; then
		find "$output_root" -mindepth 1 -maxdepth 2 -type d -empty -delete
	fi

	# Only skip when actual result files exist (in base dir or one-level subdirs).
	if [[ -d "$base_dir" ]]; then
		if find "$base_dir" -maxdepth 2 -type f \( -name "predictions.csv" -o -name "metrics.json" \) | grep -q .; then
			return 0
		fi
	fi

	return 1
}

if [[ ! -f "$eval_script" ]]; then
	echo "Error: evaluate script not found at $eval_script"
	exit 1
fi

echo "Dataset            : $dataset"
echo "Data dir           : ${data_dir:-(none)}"
echo "Press              : $press_name"
echo "Model              : $model"
echo "Devices            : $gpu_ids"
echo "Ratios             : ${compression_ratios[*]}"
echo

for ratio in "${compression_ratios[@]}"; do
	if has_existing_results "$dataset" "$data_dir" "$model" "$press_name" "$ratio"; then
		echo "[$(date '+%F %T')] Skip compression_ratio=$ratio (existing results found)"
		continue
	fi

	echo "[$(date '+%F %T')] Running compression_ratio=$ratio"
	eval_args=(
		--dataset "$dataset"
		--model "$model"
		--press_name "$press_name"
		--compression_ratio "$ratio"
	)
	if [[ -n "$data_dir" ]]; then
		eval_args+=(--data_dir "$data_dir")
	fi

	if CUDA_VISIBLE_DEVICES="$gpu_ids" uv run "$eval_script" "${eval_args[@]}"; then
		:
	else
		status=$?
		if [[ "$status" -eq 130 || "$status" -eq 143 ]]; then
			echo "[$(date '+%F %T')] Interrupted during compression_ratio=$ratio, exiting immediately."
			exit "$status"
		fi
		echo "[$(date '+%F %T')] Error compression_ratio=$ratio (exit code: $status), continue to next ratio"
		continue
	fi
done

echo "All evaluations completed."
