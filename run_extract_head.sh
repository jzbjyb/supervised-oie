javac=/home/ziyux/src/jdk1.8.0_191/bin/javac
java=/home/ziyux/src/jdk1.8.0_191/bin/java
cp=/home/zhengbaj/lib/stanford-corenlp-full-2018-02-27/*
$javac -cp "$cp" ExtractHead.java
$java -Xmx4g -cp "$cp:." ExtractHead \
	supervised-oie-benchmark/oie_corpus/test.oie.orig.correct \
	supervised-oie-benchmark/oie_corpus/test.oie.orig.correct.head &&
$java -Xmx4g -cp "$cp:." ExtractHead \
	external_datasets/mesquita_2013/processed/web.oie.correct \
	external_datasets/mesquita_2013/processed/web.oie.correct.head &&
$java -Xmx4g -cp "$cp:." ExtractHead \
	external_datasets/mesquita_2013/processed/nyt.oie.correct \
	external_datasets/mesquita_2013/processed/nyt.oie.correct.head &&
$java -Xmx4g -cp "$cp:." ExtractHead \
	external_datasets/mesquita_2013/processed/penn.oie.correct \
	external_datasets/mesquita_2013/processed/penn.oie.correct.head