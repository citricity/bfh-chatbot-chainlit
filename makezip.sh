#!/bin/bash
scriptDir=$(realpath $(dirname $0))
baseDir=$(basename $scriptDir)
zipName="${baseDir}.zip"
echo "Zipping to ${zipName}"
cd ../
zip -r6 $zipName $baseDir  -x="*__pycache__*" -x="*.git*" -x="*DS_Store*" -x="*.chainlit*" -x "*.files*" -x "*makezip.sh*"
cd -