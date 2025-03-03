#!/bin/bash

# デフォルト値の設定
DEFAULT_MODELS="gradtfkful_mask gradtfkful_plus"
DEFAULT_METRICS="dt RTF4mel utmos wer mcd logf0rmse pesq"
DEFAULT_HIFIGAN_FLAG=""

# ヘルプメッセージ
usage() {
    echo "使用方法: $0 [-m モデル] [-t 評価指標]"
    echo "オプション:"
    echo "  -m    モデルのリスト (スペース区切り、デフォルト: $DEFAULT_MODELS)"
    echo "  -t    評価指標のリスト (スペース区切り、デフォルト: $DEFAULT_METRICS)"
	echo "  -gv   hifigan version (Default: $DEFAULT_HIFIGAN_FLAG)"
    echo "  -h    このヘルプメッセージを表示"
    exit 1
}

# コマンドライン引数の処理
while getopts "m:t:gv:h" opt; do
    case ${opt} in
        m )
            MODELS=$OPTARG
            ;;
        t )
            METRICS=$OPTARG
            ;;
		gv )
			HIFIGAN_FLAG=$OPTARG
			;;
        h )
            usage
            ;;
        \? )
            usage
            ;;
    esac
done

# file selector
selector_json_path(){
	local model=$1
	local metric=$2
	local hifigan_flag=$3
	local base_path="result4eval/infer4colb/${model}/cpu/e500_n50"
	local hifigan_operand = ""

	if [ -z "${hifigan_flag}" ]; then
		hifigan_operand = "${hifigan_flag}"
	else 
		hifigan_operand = "_${hifigan_flag}"
	fi

	if [ "$metric" = "wer" ]; then
        echo "${base_path}/eval4wer${hifigan_operand}.json"
	elif [ "$metric" = "mcd" ]; then
		echo "${base_path}/eval4mcd${hifigan_operand}.json"
	elif [ "$metric" = "logf0rmse" ]; then
		echo "${base_path}/eval4logf0rmse${hifigan_operand}.json"
	elif [ "$metric" = "pesq" ] || [ "$metric" = "stoi" ]; then
		echo "${base_path}/eval4pesq${hifigan_operand}.json"
    else
        echo "${base_path}/eval4mid${hifigan_operand}.json"
    fi
}


# デフォルト値の使用（引数が指定されていない場合）
MODELS=${MODELS:-$DEFAULT_MODELS}
METRICS=${METRICS:-$DEFAULT_METRICS}
HIFIGAN_FLAG=${HIFIGAN_FLAG:-$DEFAULT_HIFIGAN_FLAG}

# 各モデルと指標の組み合わせに対して評価を実行
for model in $MODELS; do
    echo "==================================================="
    echo "モデル: $model"
    echo "==================================================="
    
    for metric in $METRICS; do
        echo "---------------------------------------------------"
        echo "評価指標: $metric"
        
        # 指標に応じてJSONファイルを選択
        JSON_FILE=$(selector_json_path "$model" "$metric" "$HIFIGAN_FLAG")
        
        # ファイルの存在確認
        if [ ! -f "$JSON_FILE" ]; then
            echo "警告: $JSON_FILE が見つかりません。スキップします。"
            continue
        fi
        
        # 評価の実行と結果表示
        python3 wcheck_json.py -it "$metric" -p "$JSON_FILE"
        echo "---------------------------------------------------"
    done
    echo ""
done
