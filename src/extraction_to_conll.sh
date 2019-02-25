eval_from=$1
eval_to=$2

# generate conll data
pushd ../supervised-oie-benchmark
python benchmark.py --gold=./oie_corpus/test.oie.orig.correct.head --out=/dev/null \
    --tabbed=${eval_from}/oie2016.txt --predArgHeadMatch --label=${eval_to}/oie2016.txt.conll &&
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/web.oie.correct.head \
    --out=/dev/null --tabbed=${eval_from}/web.txt --predArgHeadMatchExclude --label=${eval_to}/web.txt.conll &&
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/nyt.oie.correct.head \
    --out=/dev/null --tabbed=${eval_from}/nyt.txt --predArgHeadMatch --label=${eval_to}/nyt.txt.conll &&
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/penn.oie.correct.head \
    --out=/dev/null --tabbed=${eval_from}/penn.txt --predArgHeadMatchExclude --label=${eval_to}/penn.txt.conll &&
popd
