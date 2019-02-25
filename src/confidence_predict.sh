model_dir=$1
eval_from=$2
eval_to=$3
conf=$4

# generate conll data
./extraction_to_conll.sh ${eval_from} ${eval_to}

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
