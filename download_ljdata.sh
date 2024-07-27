#!/bin/bash

# usage: chmod +x ./download_ljdata.sh && ./download_ljdata.sh


BASE_DATA_DIR=./data
DATASET_NAME=ljspeech
DATASET_DIR=${BASE_DATA_DIR}/${DATASET_NAME}

function dl_ljspeech(){
	wget -P ${1} http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
	tar -jxvf ${1}/LJSpeech-1.1.tar.bz2 -C ${1}	
	#ln -s ${1}/LJSpeech-1.1
}

if [ ! -d ${BASE_DATA_DIR} ]; then
	echo "Not found DIR for dataset."
	echo "You need ${BASE_DATA_DIR}, So make by mkdir."
	exit 1
else
	if [ ! -d ${DATASET_DIR} ]; then
		echo "Not found DIR for ${DATASET_NAME} dataset."
		echo "You need ${DATASET_DIR}, So make by mkdir."
		exit 1
	else
		if [ ! -z "$(ls ${DATASET_DIR})" ]; then	
			# -z option is to retern true if argument is null.
			# so, in this case, there is no data in the directory.
			echo "some data exist in ${DATASET_DIR}."
			echo "So, there is ${DATASET_NAME}'s data in ${DATASET_DIR}."
			echo "please check again."
			exit 1
		else
			echo "Not any data exist in ${DATASET_DIR}."
			echo "So, start downloading, ${DATASET_NAME}."
			dl_ljspeech ${DATASET_DIR}
			echo ".fin download ${DATASET_NAME}"
		fi
	fi
fi
