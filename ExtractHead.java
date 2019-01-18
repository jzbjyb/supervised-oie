import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.ie.util.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.semgraph.*;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.simple.*;
import java.util.*;
import java.io.*;


public class ExtractHead {
  //public static String text = "The company said the fastener business " + 
  //  "`` has been under severe cost pressures for some time . ''";
  public static String text = "The effect is that lawsuits that might have been barred " + 
    "because they were filed too late could proceed because of the one - year extension .";

  private static class Combo {
    public String head;
    public int depth;
    public Combo(String head, int depth) {
      this.head = head;
      this.depth = depth;
    }
  }

  public static Combo topHeadFinder(Tree node, Tree parent, HeadFinder headFinder, 
    int start, int end, List<String> went, int depth) {
    if (node.isLeaf()) { // leaf node
      int ind = went.size();
      went.add(node.toString());
      if (ind < start || ind >= end) return null;
      return new Combo(node.toString(), depth);
    } else { // non-leaf node
      List<Combo> childHeads = new ArrayList<Combo>();
      int start_ind = went.size(); // start index of current node
      for (Tree child : node.children()) {
        Combo childHead = topHeadFinder(child, node, headFinder, start, end, went, depth + 1);
        if (childHead != null) childHeads.add(childHead);
      }
      int end_ind = went.size(); // end index of current node
      if (start_ind < start || end_ind > end) { // this node covers more than what we want
        if (childHeads.size() == 0) return null;
        int minDepth = childHeads.get(0).depth;
        Combo minDepthHead = childHeads.get(0);
        for (Combo co : childHeads)
          if (co.depth <= minDepth) {
            minDepth = co.depth;
            minDepthHead = co;
          }
        return minDepthHead;
      } else {
        // this node covers a subset of [start, end)
        return new Combo(node.headTerminal(headFinder, parent).toString(), depth);
      }
    }
  }

  public static void dfs(Tree node, Tree parent, HeadFinder headFinder) {
    if (node == null || node.isLeaf()) return;
    //if node is a NP - Get the terminal nodes to get the words in the NP      
    if (node.value().equals("NP")) {
      System.out.println(" Noun Phrase is ");
      List<Tree> leaves = node.getLeaves();
      for (Tree leaf : leaves)
        System.out.print(leaf.toString()+" ");
      System.out.println();
      System.out.println(" Head string is ");
      System.out.println(node.headTerminal(headFinder, parent));
    }
    for (Tree child : node.children()) {
      dfs(child, node, headFinder);
    }
  }

  public static int[] getRange(String sentence, String subSentence) {
    // words in sentence and subSentence are separated by one whitespace
    int[] result = new int[2];
    int startInd = (" " + sentence + " ").indexOf(" " + subSentence + " ");
    if (startInd < 0) 
      throw new IllegalArgumentException("can't find\n" + subSentence + "\nin\n" + sentence);
    int count = 0;
    for (int i = 0; i < startInd; i++)
      if (sentence.charAt(i) == ' ') count++;
    result[0] = count;
    count = 1;
    for (int i = 0; i < subSentence.length(); i++)
      if (subSentence.charAt(i) == ' ') count++;
    result[1] = result[0] + count;
    return result;
  }

  public static void converter(String filename, String outFilename, boolean useSimple)
    throws Exception {
    Scanner scan = new Scanner(new File(filename));
    PrintWriter pw = new PrintWriter(new FileWriter(outFilename));
    CollinsHeadFinder headFinder = new CollinsHeadFinder();
    Map<String, Tree> cache = new HashMap<String, Tree>();
    String line = null;
    StanfordCoreNLP coreNLP = null;
    if (!useSimple) {
      // initialize stanford nlp
      Properties props = new Properties();
      props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,depparse");
      coreNLP = new StanfordCoreNLP(props);
    }
    do {
      // read one line
      line = scan.nextLine();
      List<String> data = Arrays.asList(line.split("\t"));
      String sent = data.get(0);
      int confidence = 1;
      // parse the sentence
      if (!cache.containsKey(sent)) {
        if (useSimple) {
          Sentence simSent = new Sentence(Arrays.asList(sent.split(" ")));
          cache.put(sent, simSent.parse());
        } else {
          CoreDocument doc = new CoreDocument(sent);
          coreNLP.annotate(doc);
          cache.put(sent, doc.sentences().get(0).constituencyParse());
        }
      }
      Tree tree = cache.get(sent);
      // get the head for the regions we want
      List<String> heads = new ArrayList<String>();
      for (int i = 1; i < data.size(); i++) {
        int[] range = getRange(sent, data.get(i));
        List<String> went = new ArrayList<String>();
        Combo head = topHeadFinder(
          tree.children()[0], tree, headFinder, range[0], range[1], went, 0);
        if (head == null) {
          System.out.println(sent);
          System.out.println(range[0] + " " + range[1]);
          throw new Exception("can't find head");
        }
        heads.add(head.head);
      }
      if (heads.size() != data.size() - 1)
        throw new IllegalArgumentException("#head not correct");
      // output to file
      pw.println(String.join("\t", data) + "\t<SYN_HEAD>\t" + String.join("\t", heads));
    } while (scan.hasNext());
    scan.close();
    pw.close();
  }

  public static void main(String[] args) 
    throws Exception {
    String inFile = args[0];
    String outFile = args[1];
    converter(inFile, outFile, true);
  }

  public static void demo(String[] args) {
    // set up pipeline properties
    Properties props = new Properties();
    // set the list of annotators to run
    props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,depparse");
    // set a property for an annotator, 
    // in this case the coref annotator is being set to use the neural algorithm
    props.setProperty("coref.algorithm", "neural");
    // build pipeline
    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
    // create a document object
    CoreDocument document = new CoreDocument(text);
    // annnotate the document
    pipeline.annotate(document);

    // second sentence
    CoreSentence sentence = document.sentences().get(0);

    // constituency parse for the second sentence
    Tree tree = sentence.constituencyParse();
    System.out.println("Example: constituency parse");
    System.out.println(tree);
    System.out.println();
    CollinsHeadFinder headFinder = new CollinsHeadFinder();
    System.out.println(">> head finder on root");
    headFinder.determineHead(tree).pennPrint(System.out);
    System.out.println(">> head finder on root with another api");
    System.out.println(tree.children()[0].headTerminal(headFinder, tree));
    System.out.println(">> head finder on all NPs");
    dfs(tree.children()[0], tree, headFinder);
    System.out.println(">> head finder with region");
    List<String> went = new ArrayList<String>();
    Combo head = topHeadFinder(tree.children()[0], tree, headFinder, 3, 6, went, 0);
    System.out.println(head.head + " " + head.depth);
    went = new ArrayList<String>();
    head = topHeadFinder(tree.children()[0], tree, headFinder, 7, 16, went, 0);
    System.out.println(head.head + " " + head.depth);
    went = new ArrayList<String>();
    head = topHeadFinder(tree.children()[0], tree, headFinder, 10, 25, went, 0);
    System.out.println(head.head + " " + head.depth);
  }
}