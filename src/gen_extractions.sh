model_file=$1
out_dir=$2

python ./trained_oie_extractor.py --model=$model_file --in=../supervised-oie-benchmark/raw_sentences/test.txt --out=${out_dir}/oie2016.txt &&
python ./trained_oie_extractor.py --model=$model_file --in=../external_datasets/mesquita_2013/processed/web.raw --out=${out_dir}/web.txt &&
python ./trained_oie_extractor.py --model=$model_file --in=../external_datasets/mesquita_2013/processed/nyt.raw --out=${out_dir}/nyt.txt &&
python ./trained_oie_extractor.py --model=$model_file --in=../external_datasets/mesquita_2013/processed/penn.raw --out=${out_dir}/penn.txt
