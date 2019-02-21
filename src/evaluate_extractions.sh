#!/bin/bash

out_dir=$1
args="${@:2}"

pushd ../supervised-oie-benchmark
echo "=== oie2016 ==="
python benchmark.py --gold=./oie_corpus/test.oie.orig.correct.head --out=/dev/null --tabbed=${out_dir}/oie2016.txt --predArgHeadMatch $args
echo "=== web ==="
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/web.oie.correct.head --out=/dev/null --tabbed=${out_dir}/web.txt --predArgHeadMatchExclude $args
echo "=== nyt ==="
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/nyt.oie.correct.head --out=/dev/null --tabbed=${out_dir}/nyt.txt --predArgHeadMatch $args
echo "=== penn ==="
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/penn.oie.correct.head --out=/dev/null --tabbed=${out_dir}/penn.txt --predArgHeadMatchExclude $args
popd
