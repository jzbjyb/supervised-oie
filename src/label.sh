eval_dir=$1

# generate conll data
pushd ../supervised-oie-benchmark
python benchmark.py --gold=./oie_corpus/test.oie.orig.correct.head --out=/dev/null \
    --tabbed=${eval_dir}/oie2016.txt --predArgHeadMatch --label=${eval_dir}/oie2016.txt.label --label_format=raw &&
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/web.oie.correct.head \
    --out=/dev/null --tabbed=${eval_dir}/web.txt --predArgHeadMatch --label=${eval_dir}/web.txt.label --label_format=raw &&
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/nyt.oie.correct.head \
    --out=/dev/null --tabbed=${eval_dir}/nyt.txt --predArgHeadMatch --label=${eval_dir}/nyt.txt.label --label_format=raw &&
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/penn.oie.correct.head \
    --out=/dev/null --tabbed=${eval_dir}/penn.txt --predArgHeadMatch --label=${eval_dir}/penn.txt.label --label_format=raw &&
popd
