#!/bin/bash

if [[ $# == 0 || $1 == -h || $1 == --help ]]; then
  echo "Usage: zip_submission.sh RESULT_DIR"
  exit
fi

cwd=$(pwd)
result_dir=$1

cd $result_dir/preds/test
rm ../../test_submission.zip 2>/dev/null
zip -r ../../test_submission.zip . -i '*_pred_test.npy'
cd $cwd
