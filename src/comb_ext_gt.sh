eval_dir=$1

# generate conll data
pushd ../supervised-oie-benchmark
python benchmark.py --gold=./oie_corpus/test.oie.orig.correct.head.reorder --out=${eval_dir}/oie2016.txt.gt \
    --tabbed=${eval_dir}/oie2016.txt.label &&
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/web.oie.correct.head.reorder \
    --out=${eval_dir}/web.txt.gt --tabbed=${eval_dir}/web.txt.label &&
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/nyt.oie.correct.head.reorder \
    --out=${eval_dir}/nyt.txt.gt --tabbed=${eval_dir}/nyt.txt.label &&
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/penn.oie.correct.head.reorder \
    --out=${eval_dir}/penn.txt.gt --tabbed=${eval_dir}/penn.txt.label &&
popd
