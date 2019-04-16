#!/bin/bash

out_dir=$1
args="${@:2}"

match_nary="--predArgHeadMatch"
match_binary="--predArgHeadMatchExclude"

pushd ../supervised-oie-benchmark
echo "=== oie2016 ==="
python benchmark.py --gold=./oie_corpus/test.oie.orig.correct.head --out=/dev/null --tabbed=${out_dir}/oie2016.txt ${match_nary} $args
echo "=== web ==="
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/web.oie.correct.head --out=/dev/null --tabbed=${out_dir}/web.txt ${match_binary} $args
echo "=== nyt ==="
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/nyt.oie.correct.head --out=/dev/null --tabbed=${out_dir}/nyt.txt ${match_nary} $args
echo "=== penn ==="
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/penn.oie.correct.head --out=/dev/null --tabbed=${out_dir}/penn.txt ${match_binary} $args
popd
