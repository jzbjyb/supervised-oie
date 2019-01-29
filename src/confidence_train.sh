tag_model_dir=$1
train_data_dir=$2
split=$3 # train or dev
conf=$4

# generate extractions with position information
python ./trained_oie_extractor.py --model=${tag_model_dir} \
    --in=../supervised-oie-benchmark/raw_sentences/${split}.txt \
    --out=${train_data_dir}/oie2016.${split}.txt --beam=1 &&

# generate conll data for training
pushd ../supervised-oie-benchmark
python benchmark.py --gold=./oie_corpus/${split}.oie.orig.correct.head \
    --out=/dev/null --tabbed=${train_data_dir}/oie2016.${split}.txt \
    --predArgHeadMatch --label=${train_data_dir}/oie2016.${split}.txt.conll &&
popd

# train confidence model
python ./rnn/model.py --train=${train_data_dir}/oie2016.${split}.txt.conll \
    --dev=${train_data_dir}/oie2016.dev.txt.conll --test=../data/test.oie.conll \
    --load_hyperparams=${conf} --restorefrom=${tag_model_dir}
