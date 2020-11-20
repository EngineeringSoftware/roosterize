#!/bin/bash

_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )


# This script is for pre-training the models


function process_data() {
        local ds=$1; shift

        ( cd $_DIR/python/
          rm -rf $_DIR/output/$ds/data
          python -m roosterize.main extract_data_from_corpus\
                 --corpus=$_DIR/../math-comp-corpus\
                 --output=$_DIR/output/$ds/data\
                 --groups=$ds
        )
}

function train_model() {
        local ds=$1; shift
        
        ( cd $_DIR/python/
          rm -rf $_DIR/output/$ds/model
          python -m roosterize.main train_model\
                 --train=$_DIR/output/$ds/data/$ds-train\
                 --val=$_DIR/output/$ds/data/$ds-val\
                 --model-dir=$_DIR/output/$ds/model\
                 --output=$_DIR/output/$ds/data\
                 --config-file=$_DIR/configs/Stmt+ChopKnlTree+attn+copy.json
        )
}

function package_model() {
        local ds=$1; shift
        
        ( cd $_DIR/output/$ds/
          tar czf roosterize-model-$ds.tgz model/
        )
}

function eval_model() {
        local ds=$1; shift
        
        ( cd $_DIR/python/
          rm -rf $_DIR/output/$ds/results
          python -m roosterize.main eval_model\
                 --data=$_DIR/output/$ds/data/$ds-test\
                 --model-dir=$_DIR/output/$ds/model\
                 --output=$_DIR/output/$ds/results
        )
}

function retrain_all_models() {
        for ds in t1 ta; do
                process_data $ds
                train_model $ds
                package_model $ds
        done
}


# ==========
# Main function -- program entry point
# This script can be executed as ./run.sh the_function_to_run

function main() {
        local action=${1:?Need Argument}; shift

        ( cd ${_DIR}
          $action "$@"
        )
}

main "$@"

