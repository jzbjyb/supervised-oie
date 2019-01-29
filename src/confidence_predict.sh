model_dir=$1
eval_from=$2
eval_to=$3
conf=$4

# generate conll data
pushd ../supervised-oie-benchmark
python benchmark.py --gold=./oie_corpus/test.oie.orig.correct.head --out=/dev/null \
    --tabbed=${eval_from}/oie2016.txt --predArgHeadMatch --label=${eval_to}/oie2016.txt.conll &&
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/web.oie.correct.head \
    --out=/dev/null --tabbed=${eval_from}/web.txt --predArgHeadMatch --label=${eval_to}/web.txt.conll &&
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/nyt.oie.correct.head \
    --out=/dev/null --tabbed=${eval_from}/nyt.txt --predArgHeadMatch --label=${eval_to}/nyt.txt.conll &&
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/penn.oie.correct.head \
    --out=/dev/null --tabbed=${eval_from}/penn.txt --predArgHeadMatch --label=${eval_to}/penn.txt.conll &&
popd

# generate confidence score
python ./rnn/model.py --test=${eval_to}/oie2016.txt.conll:${eval_from}/oie2016.txt:${eval_to}/oie2016.txt \
    --load_hyperparams=${conf} --pretrained=${model_dir} &&
python ./rnn/model.py --test=${eval_to}/web.txt.conll:${eval_from}/web.txt:${eval_to}/web.txt \
    --load_hyperparams=${conf} --pretrained=${model_dir} &&
python ./rnn/model.py --test=${eval_to}/nyt.txt.conll:${eval_from}/nyt.txt:${eval_to}/nyt.txt \
    --load_hyperparams=${conf} --pretrained=${model_dir} &&
python ./rnn/model.py --test=${eval_to}/penn.txt.conll:${eval_from}/penn.txt:${eval_to}/penn.txt \
    --load_hyperparams=${conf} --pretrained=${model_dir} &&

# evaluate new confidence score
./evaluate_extractions.sh ${eval_to}
