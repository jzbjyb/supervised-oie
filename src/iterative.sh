niter=$1 # number of iterations
tag_model_dir=$2 # tagging model dir (the initial model should be in tag_model_dir\iter0)
train_data_dir=$3 # training data dir (the initial data should be
                  # train_data_dir\iter0\oie2016.[train|dev].iter.conll)
eval_dir=$4 # evaluation data dir
conf=$5 # configuration file
beam=$6

for (( e=1; e<=$niter; e++ ))
do
    echo "=== Iter $e ==="

    cur_dir=iter$((e-1))
    next_dir=iter${e}

    # create dir holding training data for next iter
    echo "beam search"
    mkdir -p ${train_data_dir}/${next_dir}

    for split in train dev
    do
        # beam search to generate extraction
        python ./trained_oie_extractor.py --model=${tag_model_dir}/${cur_dir} \
            --in=../supervised-oie-benchmark/raw_sentences/${split}.txt \
            --out=${train_data_dir}/${next_dir}/oie2016.${split}.beam --beam=${beam} &&
        # generate conll data for training
        pushd ../supervised-oie-benchmark
        python benchmark.py --gold=./oie_corpus/${split}.oie.orig.correct.head \
            --out=/dev/null --tabbed=${train_data_dir}/${next_dir}/oie2016.${split}.beam \
            --predArgHeadMatch --label=${train_data_dir}/${next_dir}/oie2016.${split}.beam.conll &&
        popd
    done

    # combine newly generate extractions with old ones and ground truth
    echo "combine conll"
    pushd ../supervised-oie-benchmark
    for split in train dev
    do
        last_iter_conll=${train_data_dir}/${cur_dir}/oie2016.${split}.iter.conll
        cur_conll=${train_data_dir}/${next_dir}/oie2016.${split}.beam.conll
        this_iter_conll=${train_data_dir}/${next_dir}/oie2016.${split}.iter.conll
        gold_conll=../data/${split}.oie.conll
        python combine_conll.py -inp=${last_iter_conll}:${cur_conll} -out=${this_iter_conll} &&
        python combine_conll.py -inp=${this_iter_conll}:${cur_conll} -gold=${gold_conll} \
            -out=${train_data_dir}/${next_dir}/oie2016.${split}.txt.conll
    done
    popd

    # create dir holding model for next iter
    echo "fine tune"
    mkdir -p ${tag_model_dir}/${next_dir}

    # fine tune the model
    python ./rnn/model.py --train=${train_data_dir}/${next_dir}/oie2016.train.txt.conll \
        --dev=${train_data_dir}/${next_dir}/oie2016.dev.txt.conll --test=../data/test.oie.conll \
        --load_hyperparams=${conf} --restorefrom=${tag_model_dir}/${cur_dir} \
        --saveto=${tag_model_dir}/${next_dir} &&

    # copy model.json
    cp ${tag_model_dir}/${cur_dir}/model.json ${tag_model_dir}/${next_dir}/.

    # evaluate the tagging model
    echo "evaluate"
    mkdir -p ${eval_dir}/${next_dir}
    ./gen_extractions.sh ${tag_model_dir}/${next_dir}/ ${eval_dir}/${next_dir} &&
    ./evaluate_extractions.sh ${eval_dir}/${next_dir}
done
