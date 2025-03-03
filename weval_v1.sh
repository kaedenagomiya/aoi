#!/bin/bash

# デフォルト値の設定
DEFAULT_MODELS="gradtfkful_mask gradtfkful_plus"
DEFAULT_METRICS="wer dt RTF4mel utmos logf0rmse"
DEFAULT_HIFIGAN=""  # HiFiGANのデフォルトバージョン

# JSONファイルのパスを選択する関数
selector_json_path() {
    local model=$1
    local metric=$2
    local base_path="result4eval/infer4colb/${model}/cpu/e500_n50"
    
    # HiFiGANバージョンの処理
    if [ -z "${hifigan_flag}" ]; then
        hifigan_operand=""
    else
        hifigan_operand="_${hifigan_flag}"
    fi
    
    if [ "$metric" = "wer" ]; then
        echo "${base_path}/eval4wer${hifigan_operand}.json"
    elif [ "$metric" = "mcd" ]; then
        echo "${base_path}/eval4mcd${hifigan_operand}.json"
    elif [ "$metric" = "logf0rmse" ]; then
        echo "${base_path}/eval4logf0rmse${hifigan_operand}.json"
    elif [ "$metric" = "pesq" ] || [ "$metric" = "stoi" ] || [ "$metric" = "estoi" ]; then
        echo "${base_path}/eval4pesq${hifigan_operand}.json"
    else
        echo "${base_path}/eval4mid${hifigan_operand}.json"
    fi
}

# ヘルプメッセージ
usage() {
    echo "使用方法: $0 [-m モデル] [-t 評価指標] [-g HiFiGANバージョン]"
    echo "オプション:"
    echo "  -m    モデルのリスト (スペース区切り、デフォルト: $DEFAULT_MODELS)"
    echo "  -t    評価指標のリスト (スペース区切り、デフォルト: $DEFAULT_METRICS)"
    echo "  -g    HiFiGANバージョン (デフォルト: $DEFAULT_HIFIGAN、'none'で無効化)"
    echo "  -h    このヘルプメッセージを表示"
    exit 1
}

# コマンドライン引数の処理
while getopts "m:t:g:h" opt; do
    case ${opt} in
        m )
            MODELS=$OPTARG
            ;;
        t )
            METRICS=$OPTARG
            ;;
        g )
            hifigan_flag=$OPTARG
            ;;
        h )
            usage
            ;;
        \? )
            usage
            ;;
    esac
done

# デフォルト値の使用（引数が指定されていない場合）
MODELS=${MODELS:-$DEFAULT_MODELS}
METRICS=${METRICS:-$DEFAULT_METRICS}
# HiFiGANフラグのデフォルト値設定（'none'の場合は空文字列に）
if [ -z "${hifigan_flag}" ]; then
    hifigan_flag=$DEFAULT_HIFIGAN
fi
if [ "${hifigan_flag}" = "none" ]; then
    hifigan_flag=""
fi

# 各モデルと指標の組み合わせに対して評価を実行
for model in $MODELS; do
    echo "==================================================="
    echo "モデル: $model"
    if [ ! -z "${hifigan_flag}" ]; then
        echo "HiFiGANバージョン: ${hifigan_flag}"
    else
        echo "HiFiGANバージョン: なし"
    fi
    echo "==================================================="
    
    for metric in $METRICS; do
        echo "---------------------------------------------------"
        echo "評価指標: $metric"
        
        # 関数を使用してJSONファイルのパスを取得
        JSON_FILE=$(selector_json_path "$model" "$metric")
        
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
