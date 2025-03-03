#!/bin/bash

# ex.
# full
# > bash wcheck.sh -m "gradtts,gradseptts,gradtfktts,gradtfk5tts,gradtfkful_plus,gradtfkful_mask,gradtimektts,gradfreqktts" -t "RTF4mel,utmos,wer,mcd,logf0rmse,pesq,stoi,estoi" -v "LJ_V1" -f "csv"
# short
# > bash wcheck.sh -m "gradtts" -t "RTF4mel,mcd,logf0rmse,wer,pesq,stoi,estoi,utmos" -v "LJ_V1" -f "csv"


# デフォルト値の設定
MODELS=("gradtts" "gradtfktts")
METRICS=("wer" "dt" "RTF4mel" "utmos")
BASE_DIR="result4eval/infer4colb"
CPU_DIR="cpu/e500_n50"
HIFIGAN_VERSION="LJ_V1"  # デフォルトのHiFi-GANバージョン
OUTPUT_FORMAT="normal" # デフォルトの出力形式

# 使用方法の表示関数
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -m, --models    Specify models (comma-separated)"
    echo "  -t, --metrics   Specify metrics (comma-separated)"
    echo "  -v, --hifigan   Specify HiFi-GAN version (default: v1)"
    echo "  -f, --format    Specify output format (normal/csv) (default: normal)"
    echo "  -h, --help      Show this help message"
    echo ""
    echo "Available models: ${MODELS[*]}"
    echo "Available metrics: ${METRICS[*]}"
}

selector_json_path() {
	local model=$1
    local metric=$2
	local base_path="${BASE_DIR}/${model}/${CPU_DIR}"

	# HiFiGANバージョンの処理
    if [ -z "${HIFIGAN_VERSION}" ]; then
        hifigan_operand=""
    else
        hifigan_operand="_${HIFIGAN_VERSION}"
    fi

	if [ "$metric" = "dt" ] || [ "$metric" = "RTF4mel" ]; then
		echo "${base_path}/eval4mid.json"
	elif [ "$metric" = "wer" ]; then
        echo "${base_path}/eval4wer${hifigan_operand}.json"
    elif [ "$metric" = "mcd" ]; then
        echo "${base_path}/eval4mcd${hifigan_operand}.json"
    elif [ "$metric" = "logf0rmse" ]; then
        echo "${base_path}/eval4logf0rmse${hifigan_operand}.json"
    elif [ "$metric" = "pesq" ] || [ "$metric" = "stoi" ] || [ "$metric" = "estoi" ]; then
        echo "${base_path}/eval4pesq${hifigan_operand}.json"
    elif [ "$metric" = "dnsmos" ] || [ "$metric" = "dnsovrl" ] || [ "$metric" = "dnssig" ] || [ "$metric" = "dnsbak" ]; then
        echo "${base_path}/eval4dnsmos${hifigan_operand}.json"
    else
        echo "${base_path}/eval4mid${hifigan_operand}.json"
    fi
}


# コマンドライン引数の処理
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--models)
            IFS=',' read -ra MODELS <<< "$2"
            shift 2
            ;;
        -t|--metrics)
            IFS=',' read -ra METRICS <<< "$2"
            shift 2
            ;;
        -v|--hifigan)
            HIFIGAN_VERSION="$2"
            shift 2
            ;;
        -f|--format)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# 結果を保存する配列
declare -A means
declare -A stds

# 結果を収集
for model in "${MODELS[@]}"; do
    [[ "$OUTPUT_FORMAT" == "normal" ]] && echo "Processing model: $model"
    
    for metric in "${METRICS[@]}"; do
        [[ "$OUTPUT_FORMAT" == "normal" ]] && echo "  Evaluating metric: $metric"

		file_path=$(selector_json_path "$model" "$metric")
		#echo $file_path
        
        # ファイルの存在確認
        if [[ ! -f "$file_path" ]]; then
            [[ "$OUTPUT_FORMAT" == "normal" ]] && echo "    Warning: File not found - $file_path"
            means["${model}_${metric}"]="N/A"
            stds["${model}_${metric}"]="N/A"
            continue
        fi
        
        # python3スクリプトを実行し結果を取得
        output=$(python3 wcheck_json.py -it "$metric" -p "$file_path")
        
        # 平均値と標準偏差を抽出
        mean=$(echo "$output" | grep "mean:" | awk '{print $NF}')
        std=$(echo "$output" | grep "std:" | awk '{print $NF}')
        
        # 結果を保存
        means["${model}_${metric}"]=$mean
        stds["${model}_${metric}"]=$std
        
        # 通常フォーマットの場合、詳細な出力を表示
        if [[ "$OUTPUT_FORMAT" == "normal" ]]; then
            echo "    Results for $metric:"
            echo "      Mean: $mean"
            echo "      Std:  $std"
        fi
    done
    [[ "$OUTPUT_FORMAT" == "normal" ]] && echo ""
done

# CSV形式での出力
if [[ "$OUTPUT_FORMAT" == "csv" ]]; then
    # ヘッダー行の出力
    echo -n "Model"
    for metric in "${METRICS[@]}"; do
        echo -n ",$metric"
    done
    echo
    
    # 各モデルの結果を mean±std の形式で出力
    for model in "${MODELS[@]}"; do
        echo -n "$model"
        for metric in "${METRICS[@]}"; do
            mean="${means[${model}_${metric}]}"
            std="${stds[${model}_${metric}]}"
            
            if [[ "$mean" == "N/A" || "$std" == "N/A" ]]; then
                echo -n ",N/A"
            else
                # 数値を小数点以下4桁に整形
                mean=$(printf "%.7f" $mean)
                std=$(printf "%.7f" $std)
                #echo -n ",$mean±$std"
				#echo -n ",$mean\$\\pm\$$std"
				#echo -n ",$mean,$std"
				echo -n ",$mean"
            fi
        done
        echo
    done

# 通常フォーマットでの詳細な出力
elif [[ "$OUTPUT_FORMAT" == "normal" ]]; then
    echo "Summary of Results:"
    echo "=================="
    
    # 各メトリクスについての詳細な統計情報を表示
    for metric in "${METRICS[@]}"; do
        echo "Metric: $metric"
        echo "--------------------"
        
        for model in "${MODELS[@]}"; do
            mean="${means[${model}_${metric}]}"
            std="${stds[${model}_${metric}]}"
            
            if [[ "$mean" == "N/A" || "$std" == "N/A" ]]; then
                echo "$model: N/A"
            else
                mean=$(printf "%.4f" $mean)
                std=$(printf "%.4f" $std)
                echo "$model: $mean ± $std"
				#echo -n ",$mean\$\\pm\$$std"
				#echo "$model: $mean"
            fi
        done
        echo ""
    done
fi
