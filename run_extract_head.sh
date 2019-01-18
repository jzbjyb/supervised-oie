javac=/home/ziyux/src/jdk1.8.0_191/bin/javac
java=/home/ziyux/src/jdk1.8.0_191/bin/java
$javac -cp "/home/zhengbaj/lib/stanford-corenlp-full-2018-02-27/*" ExtractHead.java
$java -Xmx4g -cp "/home/zhengbaj/lib/stanford-corenlp-full-2018-02-27/*:." ExtractHead \
	supervised-oie-benchmark/oie_corpus/test.oie.orig.correct \
	supervised-oie-benchmark/oie_corpus/test.oie.orig.correct.head.test