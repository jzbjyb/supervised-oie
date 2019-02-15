out_dir=$1
args=$2

./evaluate_extractions.sh $out_dir $args |& grep -P "^ \." | sed ':a;N;$!ba;s/\n/\t/g'
