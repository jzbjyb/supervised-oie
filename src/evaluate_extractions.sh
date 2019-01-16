out_dir=$1

pushd ../supervised-oie-benchmark
echo "=== oie2016 ==="
python benchmark.py --gold=./oie_corpus/test.oie.orig --out=/dev/null --tabbed=${out_dir}/oie2016.txt --predArgMatch
echo "=== web ==="
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/web.oie --out=/dev/null --tabbed=${out_dir}/web.txt --predArgMatch
echo "=== nyt ==="
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/nyt.oie --out=/dev/null --tabbed=${out_dir}/nyt.txt --predArgMatch
echo "=== penn ==="
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/penn.oie --out=/dev/null --tabbed=${out_dir}/penn.txt --predArgMatch
popd
