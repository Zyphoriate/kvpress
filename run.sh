#!/usr/bin/env bash

set -euo pipefail

on_interrupt() {
	echo
	echo "[$(date '+%F %T')] Interrupted by Ctrl+C, exiting immediately."
	exit 130
}

trap on_interrupt INT TERM

# Usage:
#   ./run.sh <dataset/data_dir> <press_name> <gpu_ids_csv> [model] [kwargs...]
# Example:
#   ./run.sh infinitebench/passkey chunckkv 2,3,4,5
#   ./run.sh truthful_qa finch 5 --query_aware true

if [[ $# -lt 3 ]]; then
	echo "Usage: $0 <dataset/data_dir> <press_name> <gpu_ids_csv> [model] [kwargs...]"
	exit 1
fi

target="$1"
press_name="$2"
gpu_ids="$3"
shift 3

# Determine if the next arg is a model name or a kwarg
model="Qwen/Qwen3-8B"
extra_args=()
if [[ $# -gt 0 && "$1" != -* ]]; then
	model="$1"
	shift
fi
# Everything remaining is passed through as kwargs
extra_args=("$@")

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
compression_ratios=(0 0.25 0.5 0.7 0.8 0.85)

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eval_script="$script_dir/evaluation/evaluate.py"
output_root="$script_dir/results"

has_existing_results() {
	local dataset="$1"
	local data_dir="$2"
	local model="$3"
	local press_name="$4"
	local ratio="$5"
	shift 5
	# Remaining positional args are extra kwargs passed through to evaluate.py

	local ratio_fmt
	ratio_fmt=$(printf "%.2f" "$ratio")
	local model_tag="${model//\//--}"

	# Build the base directory name the same way get_results_dir() does in evaluate.py
	# See EvaluationConfig.get_results_dir() for the canonical logic.
	local components=()
	components+=("$dataset")
	if [[ -n "$data_dir" ]]; then
		components+=("$data_dir")
	fi
	components+=("$model_tag")
	components+=("$press_name")

	# Parse extra kwargs that affect the results directory name.
	# Collect values first, then append in the fixed order used by get_results_dir().
	local fraction_val=""
	local max_ctx_val=""
	local query_aware_flag=""
	local key_channel_cr_val=""
	local thresh_val=""
	local nd_val=""

	local args=("$@")
	local i=0
	while [[ $i -lt ${#args[@]} ]]; do
		case "${args[$i]}" in
			--fraction)            fraction_val="${args[$((i+1))]}"; i=$((i+2));;
			--max_context_length)  max_ctx_val="${args[$((i+1))]}"; i=$((i+2));;
			--query_aware)         query_aware_flag="${args[$((i+1))]}"; i=$((i+2));;
			--key_channel_compression_ratio) key_channel_cr_val="${args[$((i+1))]}"; i=$((i+2));;
			--threshold)           thresh_val="${args[$((i+1))]}"; i=$((i+2));;
			--needle_depth)        nd_val="${args[$((i+1))]}"; i=$((i+2));;
			*)                     i=$((i+1));;
		esac
	done

	# Append in the exact order used by get_results_dir()
	if [[ -n "$thresh_val" ]]; then
		# threshold replaces compression_ratio
		local thresh_fmt
		thresh_fmt=$(printf "%.2f" "$thresh_val")
		components+=("$thresh_fmt")
	else
		components+=("$ratio_fmt")
	fi

	if [[ -n "$fraction_val" && "$fraction_val" != "1.0" && "$fraction_val" != "1" ]]; then
		local frac_fmt
		frac_fmt=$(printf "fraction%.3f" "$fraction_val")
		components+=("$frac_fmt")
	fi

	if [[ -n "$max_ctx_val" ]]; then
		components+=("max_context${max_ctx_val}")
	fi

	if [[ "$query_aware_flag" == "true" ]]; then
		components+=("query_aware")
	fi

	if [[ -n "$key_channel_cr_val" ]]; then
		local kc_fmt
		kc_fmt=$(printf "key_channel_cr%.2f" "$key_channel_cr_val")
		components+=("$kc_fmt")
	fi

	if [[ -n "$nd_val" && "$dataset" == "needle_in_haystack" ]]; then
		# Python str(list) inserts spaces after commas: "[0, 25, 50]"
		# Normalize the shell value (compact "[0,25,50]") to match.
		if [[ "$nd_val" == [* ]]; then
			nd_val="${nd_val//, /,}"
			nd_val="${nd_val//,/, }"
		fi
		components+=("needle_depth${nd_val}")
	fi

	local result_dir_name
	result_dir_name=$(printf '%s__' "${components[@]}")
	result_dir_name="${result_dir_name%__}"  # strip trailing __

	local base_dir="$output_root/$result_dir_name"

	# Remove empty subdirectories before checking (leave files untouched).
	if [[ -d "$base_dir" ]]; then
		find "$base_dir" -mindepth 1 -maxdepth 5 -type d -empty -delete
		rmdir --ignore-fail-on-non-empty "$base_dir"
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
if [[ ${#extra_args[@]} -gt 0 ]]; then
	echo "Extra args         : ${extra_args[*]}"
fi
echo

for ratio in "${compression_ratios[@]}"; do
	if has_existing_results "$dataset" "$data_dir" "$model" "$press_name" "$ratio" "${extra_args[@]}"; then
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

	# Append any extra kwargs passed from the command line
	if [[ ${#extra_args[@]} -gt 0 ]]; then
		eval_args+=("${extra_args[@]}")
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
