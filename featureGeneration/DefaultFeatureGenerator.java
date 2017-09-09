package edu.washington.multirframework.featuregeneration;

import java.util.ArrayList;
import java.util.List;
//import java.util.Map;
import java.util.Properties;

import java.io.BufferedReader;
//import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

//import java.lang.*;

import edu.stanford.nlp.dcoref.CorefChain;
import edu.stanford.nlp.dcoref.CorefCoreAnnotations.CorefChainAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.BasicDependenciesAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.util.CoreMap;

/**
 * Default implementation of FeatureGenerator
 * inspired by the Mintz features, but slightly
 * different.
 * @author jgilme1
 *
 */
public class DefaultFeatureGenerator implements FeatureGenerator {


  public static void main(String[] args) {
    // creates a StanfordCoreNLP object, with POS tagging, lemmatization, NER, parsing, and coreference resolution 
      for (String arg: args) {
        System.out.println("one arg: " + arg);
      }
      Properties props = new Properties();

      //Natalie: example properties from CoreNLPProcessor.java (in bitbucket relation-extraction project)
      //properties.put("annotators", "tokenize, ssplit, pos, lemma, ner");

      props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");
      StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
      try {
        // input file: args[0]
        // current input format: std file
        File file = new File(args[0]);
        FileReader fileReader = new FileReader(file);
        BufferedReader bufferedReader = new BufferedReader(fileReader);
        
        // output file: args[1]
        // current output format: std file
        FileWriter fileWriter = new FileWriter(args[1]);

        
        int count = 0;
        String line;
        while ((line = bufferedReader.readLine()) != null) {
          String[] parts = line.split("\t");
          System.out.println("length: " + parts.length);
          // session id
          String sessionId = parts[0];
          // arg1
          String arg1 = parts[1]; 
          // arg1 start offset
          int arg1StartOffset = Integer.parseInt(parts[2]); 
          // arg1 end offset
          int arg1EndOffset = Integer.parseInt(parts[3]);
          // arg2
          String arg2 = parts[4]; 
          // arg2 start offset
          int arg2StartOffset = Integer.parseInt(parts[5]);
          // arg2 end offset
          int arg2EndOffset = Integer.parseInt(parts[6]); 
          // sentence id
          String sentenceId = parts[7];
          // relation
          String relation = parts[8]; 
          // sentence
          String text = parts[9];
          System.out.println("text: " + text);
          
          
          // create an empty Annotation just with the given text
          Annotation document = new Annotation(text);
          // run all Annotators on this text
          pipeline.annotate(document);
          // these are all the sentences in this document
          List<CoreMap> sentences = document.get(SentencesAnnotation.class);
          for (CoreMap map: sentences) {
            System.out.println("map: " + map.toString());
          }
          // feature generator
          DefaultFeatureGenerator fg = new DefaultFeatureGenerator();
          
          
          List<String> features = null;
          try {
            features = fg.generateFeatures(arg1StartOffset,arg1EndOffset,arg2StartOffset,arg2EndOffset,"a1","a2", sentences.get(0), document);
        
            String featureVec = "";
            for (String feature:features) {
              featureVec = featureVec + feature + "\t0\t";
            }
            // truncate the last tab
            featureVec = featureVec.substring(0, featureVec.length()-1);
            
            // construct an std row
            String sentLine = sessionId + "\t" + arg1+"\t"+Integer.toString(arg1StartOffset)+"\t"+Integer.toString(arg1EndOffset)+"\t"
                +arg2+"\t"+Integer.toString(arg2StartOffset)+"\t"+Integer.toString(arg2EndOffset)+"\t['"
                +relation+"']\t"+text+"\t"+featureVec+"\n";
            
            fileWriter.write(sentLine);
            
            count = count + 1;
          
          } catch (Exception e2) {
//            System.out.println("ill-formatted feature or no roots in graph!");
            throw e2;
          }
          
          if (count % 100 == 0)
            System.out.println(count);
//          if (count == 2000)
//            break;
        }
        fileReader.close();
        fileWriter.close();
        
      } catch (IOException e) {
        e.printStackTrace(); // where stack trace got printed
      }
      
      return ;
  }


  @Override
  public List<String> generateFeatures(Integer arg1StartOffset,
      Integer arg1EndOffset, Integer arg2StartOffset,
      Integer arg2EndOffset, String arg1ID, String arg2ID, CoreMap sentence, Annotation document) {
    
    List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
    
    //initialize arguments
    String[] tokenStrings = new String[tokens.size()];
    String[] posTags = new String[tokens.size()];
    
    //initialize dependency parents to -1
    int[] depParents = new int[tokens.size()];
    for(int i = 0; i < depParents.length; i ++){
      depParents[i] = -1;
    }
    
    
    String[] depTypes = new String[tokens.size()];
    
    
    String arg1ner = "";
    String arg2ner = "";
    int[] arg1Pos = new int[2];
    int[] arg2Pos = new int[2];

    //iterate over tokens
    for(int i =0; i < tokens.size(); i++){
      
      CoreLabel token = tokens.get(i);
      
      //set the tokenString value
      tokenStrings[i] =token.value();
      
      //set the pos value
      String pos = token.get(CoreAnnotations.PartOfSpeechAnnotation.class);
      if(pos == null){
        posTags[i] = "";
      }
      else{
        posTags[i] = pos;
      }
          
    }
    
    arg1Pos[0] = arg1StartOffset;
    arg1Pos[1] = arg1EndOffset-1;
    arg1ner = tokens.get(arg1Pos[0]).get(CoreAnnotations.NamedEntityTagAnnotation.class);
    for (CoreLabel ner: tokens) {
      System.out.println("ner: " + ner.get(CoreAnnotations.NamedEntityTagAnnotation.class));
    }
    arg2Pos[0] = arg2StartOffset;
    arg2Pos[1] = arg2EndOffset-1;
    arg2ner = tokens.get(arg2Pos[0]).get(CoreAnnotations.NamedEntityTagAnnotation.class);
    

    //dependency conversions..
//    List<Triple<Integer,String,Integer>> dependencyData = sentence.get(DependencyAnnotation.class);
//    if(dependencyData != null){
//      for(Triple<Integer,String,Integer> dep : dependencyData){
    SemanticGraph graph = sentence.get(BasicDependenciesAnnotation.class);
    for (SemanticGraphEdge edge:graph.getEdgeSet()) {
      int parent = edge.getSource().get(CoreAnnotations.IndexAnnotation.class) - 1;
      String type = edge.getRelation().toString();
      int child = edge.getTarget().get(CoreAnnotations.IndexAnnotation.class) - 1;
      //System.out.println(parent+":"+type+":"+child);
//        int parent = dep.first -1;
//        String type = dep.second;
//        int child = dep.third -1;
  
        //child and parent should not be equivalent
        if(parent == child){
          parent = -1;
        }
        
        if(child < tokens.size()){
          depParents[child] = parent;
          depTypes[child] = type;
        }
        else{
          System.err.println("ERROR BETWEEN DEPENDENCY PARSE AND TOKEN SIZE");
          return new ArrayList<String>();
        }
      }
    depTypes[graph.getFirstRoot().get(CoreAnnotations.IndexAnnotation.class) - 1] = "root";
//    }
//    else{
//      return new ArrayList<String>();
//    }
//    System.err.println(depTypes[4]);
    //add 1 to end Pos values
    arg1Pos[1] += 1;
    arg2Pos[1] += 1;
  
    return originalMultirFeatures(tokenStrings, posTags, depParents, depTypes, arg1Pos, arg2Pos, arg1ner, arg2ner);
  }
  
  
   /** 
   * RelationECML getFeatures algorithm
         * @author Raphael Hoffmann
         * @modified Xiao Ling
   * @param tokens
   * @param postags
   * @param depParents
   * @param depTypes
   * @param arg1Pos
   * @param arg2Pos
   * @param arg1ner
   * @param arg2ner
   * @return
   */
  public static  List<String> originalMultirFeatures( String[] tokens, 
      String[] postags,
      int[] depParents, String[] depTypes,
      int[] arg1Pos, int[] arg2Pos, String arg1ner, String arg2ner) {
 
    List<String> features = new ArrayList<String>();
 
    // it's easier to deal with first, second
    int[] first = arg1Pos, second = arg2Pos;
    String firstNer = arg1ner, secondNer = arg2ner;
    if (arg1Pos[0] > arg2Pos[0]) {
      second = arg1Pos; first = arg2Pos;
      firstNer = arg2ner; secondNer = arg1ner;
    }
    
    // define the inverse prefix
    String inv = (arg1Pos[0] > arg2Pos[0])? 
        "inverse_true" : "inverse_false";
    
    // define the middle parts
    StringBuilder middleTokens = new StringBuilder();
    for (int i=first[1]; i < second[0]; i++) {
      if (i > first[1]) {
        middleTokens.append(" ");
      }
      middleTokens.append(tokens[i]);
    }
    
    if (second[0] - first[1] > 10) {
      middleTokens.setLength(0);
      middleTokens.append("*LONG*");
    }
    
    // define the prefixes and suffixes
    String[] prefixTokens = new String[2];
    String[] suffixTokens = new String[2];
    
    for (int i=0; i < 2; i++) {
      int tokIndex = first[0] - i - 1;
      if (tokIndex < 0) prefixTokens[i] = "B_" + tokIndex;
      else prefixTokens[i] = tokens[tokIndex];
    }
 
    for (int i=0; i < 2; i++) {
      int tokIndex = second[1] + i;
      if (tokIndex >= tokens.length) suffixTokens[i] = "B_" + (tokIndex - tokens.length + 1);
      else suffixTokens[i] = tokens[tokIndex];
    }
 
    String[] prefixes = new String[3];
    String[] suffixes = new String[3];
 
    prefixes[0] = suffixes[0] = "";
    prefixes[1] = prefixTokens[0];
    prefixes[2] = prefixTokens[1] + " " + prefixTokens[0];
    suffixes[1] = suffixTokens[0];
    suffixes[2] = suffixTokens[0] + " " + suffixTokens[1];
    
    // generate the features in the same order as in ecml data
    String mto = middleTokens.toString();
    
    features.add(inv + "|" + firstNer + "|" + mto + "|" + secondNer);
    features.add(inv + "|" + prefixes[1] + "|" + firstNer + "|" + mto + "|" + secondNer + "|" + suffixes[1]);
    features.add(inv + "|" + prefixes[2] + "|" + firstNer + "|" + mto + "|" + secondNer + "|" + suffixes[2]);
    
    
    
    
    
 
    
    // dependency features
    if (depParents == null || depParents.length < tokens.length) return features;
    
    // identify head words of arg1 and arg2
    // (start at end, while inside entity, jump)
    int head1 = arg1Pos[1]-1;
    int loopIterationCount =0;
    while (depParents[head1] >= arg1Pos[0] && depParents[head1] < arg1Pos[1]) {
        head1 = depParents[head1];
        //avoid infinite loop
        if(loopIterationCount == 100){
          break;
        }
        loopIterationCount++;
    }
    int head2 = arg2Pos[1]-1;
    //System.out.println(head1 + " " + head2);
    loopIterationCount =0;
    while (depParents[head2] >= arg2Pos[0] && depParents[head2] < arg2Pos[1]) {
      head2 = depParents[head2];
      //avoid infinite loop
        if(loopIterationCount == 100){
          break;
        }
        loopIterationCount++;
    }
    
    
    // find path of dependencies from first to second
    int[] path1 = new int[tokens.length];
    for (int i=0; i < path1.length; i++) path1[i] = -1;
    path1[0] = head1; // last token of first argument
    for (int i=1; i < path1.length; i++) {
      path1[i] = depParents[path1[i-1]];
      if (path1[i] == -1) break;
    } 
    int[] path2 = new int[tokens.length];
    for (int i=0; i < path2.length; i++) path2[i] = -1;
    path2[0] = head2; // last token of first argument
    for (int i=1; i < path2.length; i++) {
      path2[i] = depParents[path2[i-1]];
      if (path2[i] == -1) break;
    }
    int lca = -1;
    int lcaUp = 0, lcaDown = 0;
    outer:
    for (int i=0; i < path1.length; i++)
      for (int j=0; j < path2.length; j++) {
        if (path1[i] == -1 || path2[j] == -1) {
          break; // no path
        }
        if (path1[i] == path2[j]) {
          lca = path1[i];
          lcaUp = i;
          lcaDown = j;
          break outer;
        }
      }
    if (lca < 0) return features; // no dependency path (shouldn't happen)
    
    
    
    
    
    
    
    
    
    String[] dirs = new String[lcaUp + lcaDown];
    String[] strs = new String[lcaUp + lcaDown];
    String[] rels = new String[lcaUp + lcaDown];
 
    StringBuilder middleDirs = new StringBuilder();
    StringBuilder middleRels = new StringBuilder();
    StringBuilder middleStrs = new StringBuilder();
 
    if (lcaUp + lcaDown < 12) {
      
      for (int i=0; i < lcaUp; i++) {
        dirs[i] = "->";
        strs[i] = i > 0? tokens[path1[i]] : "";
        rels[i] = depTypes[path1[i]];
        //System.out.println("[" + depTypes[path1[i]] + "]->");
      }
      for (int j=0; j < lcaDown; j++) {
      //for (int j=lcaDown-1; j >= 0; j--) {
        dirs[lcaUp + j] = "<-";
        strs[lcaUp + j] = (lcaUp + j > 0)? tokens[path2[lcaDown-j]] : ""; // word taken from above
        rels[lcaUp + j] = depTypes[path2[lcaDown-j-1]];
        //System.out.println("[" + depTypes[path2[j]] + "]<-");
      }
      
      for (int i=0; i < dirs.length; i++) {
        middleDirs.append(dirs[i]);
        middleRels.append("[" + rels[i] + "]" + dirs[i]);
        middleStrs.append(strs[i] + "[" + rels[i] + "]" + dirs[i]);
      }
      
    }
    else {
        middleDirs.append("*LONG-PATH*");
        middleRels.append("*LONG-PATH*");
        middleStrs.append("*LONG-PATH*");
    }
  
    String basicDir = arg1ner + "|" + middleDirs.toString() + "|" + arg2ner;
    String basicDep = arg1ner + "|" + middleRels.toString() + "|" + arg2ner;
    String basicStr = arg1ner + "|" + middleStrs.toString() + "|" + arg2ner;
    
 
    
    
    
    
    
    
    
    
    
    // new left and right windows: all elements pointing to first arg, but not on path
    //List<Integer> lws = new ArrayList<Integer>();
    //List<Integer> rws = new ArrayList<Integer>();
    
    List<String> arg1dirs = new ArrayList<String>();
    List<String> arg1deps = new ArrayList<String>();
    List<String> arg1strs = new ArrayList<String>();
    List<String> arg2dirs = new ArrayList<String>();
    List<String> arg2deps = new ArrayList<String>();
    List<String> arg2strs = new ArrayList<String>();
    
    // pointing out of argument
    for (int i=0; i < tokens.length; i++) {
      // make sure itself is not either argument
            //if (i >= first[0] && i < first[1]) continue;
            //if (i >= second[0] && i < second[1]) continue;
      if (i == head1) continue;
      if (i == head2) continue;
      
      // make sure i is not on path
      boolean onPath = false;
      for (int j=0; j < lcaUp+1; j++) if (path1[j] == i) onPath = true;
      for (int j=0; j < lcaDown+1; j++) if (path2[j] == i) onPath = true;
      if (onPath) continue;
      // make sure i points to first or second arg
      //if (depParents[i] >= first[0] && depParents[i] < first[1]) lws.add(i);
      //if (depParents[i] >= second[0] && depParents[i] < second[1]) rws.add(i);
      if (depParents[i] == head1) {
        //lws.add(i);
        arg1dirs.add("->");       
        arg1deps.add("[" + depTypes[i] + "]->");
        arg1strs.add(tokens[i] + "[" + depTypes[i] + "]->");
      }
      if (depParents[i] == head2) {
        //rws.add(i);     
        arg2dirs.add("<-");       
        arg2deps.add("[" + depTypes[i] + "]<-");
        arg2strs.add("[" + depTypes[i] + "]<-" + tokens[i]);
      }
    }
    
    
    // case 1: pointing into the argument pair structure (always attach to lhs):
    // pointing from arguments
    if (lcaUp == 0 && depParents[head1] != -1 /*|| depParents[head1] == head2*/) {
      arg1dirs.add("<-");       
      arg1deps.add("[" + depTypes[head1] + "]<-");
      arg1strs.add(tokens[depParents[head1]] + "[" + depTypes[head1] + "]<-");
      
      if (depParents[depParents[head1]] != -1) {
        arg1dirs.add("<-");
        arg1deps.add("[" + depTypes[depParents[head1]] + "]<-");
        arg1strs.add(tokens[depParents[depParents[head1]]] + "[" + depTypes[depParents[head1]] + "]<-");
      }
    }
 
    // if parent is not on path or if parent is 
    if (lcaDown == 0 && depParents[head2] != -1 /*|| depParents[head2] == head1*/) { // should this actually attach to rhs???
      arg2dirs.add("->");
      arg2deps.add("[" + depTypes[head2] + "]->");
      arg2strs.add("[" + depTypes[head2] + "]->"+tokens[depParents[head2]]);
      
      if (depParents[depParents[head2]] != -1) {
        arg2dirs.add("->");
        arg2deps.add("[" + depTypes[depParents[head2]] + "]->");
        arg2strs.add("[" + depTypes[depParents[head2]] + "]->"+tokens[depParents[depParents[head2]]]);
      }
    }
 
    // case 2: pointing out of argument
    
    //features.add("dir:" + basicDir);    
    //features.add("dep:" + basicDep);
 
    
    // left and right, including word
    for (String w1 : arg1strs)
      for (String w2 : arg2strs)
        features.add("str:" + w1 + "|" + basicStr + "|" + w2);
    
    /*
    for (int lw : lws) {
      for (int rw : rws) {
        features.add("str:" + tokens[lw] + "[" + depTypes[lw] + "]<-" + "|" + basicStr
            + "|" + "[" + depTypes[rw] + "]->" + tokens[rw]);
      }
    }
    */
    
    
    
    // only left
    for (int i=0; i < arg1dirs.size(); i++) {
      features.add("str:" + arg1strs.get(i) + "|" + basicStr);
      features.add("dep:" + arg1deps.get(i) + "|" + basicDep);
      features.add("dir:" + arg1dirs.get(i) + "|" + basicDir);
    }
    
    
    // only right
    for (int i=0; i < arg2dirs.size(); i++) {
      features.add("str:" + basicStr + "|" + arg2strs.get(i));
      features.add("dep:" + basicDep + "|" + arg2deps.get(i));
      features.add("dir:" + basicDir + "|" + arg2dirs.get(i));
    }
 
    features.add("str:" + basicStr);
 
    return features;
  }
  
//  /** 
//   * Untouched RelationECML getFeatures algorithm...
//   * @param tokens
//   * @param postags
//   * @param depParents
//   * @param depTypes
//   * @param arg1Pos
//   * @param arg2Pos
//   * @param arg1ner
//   * @param arg2ner
//   * @return
//   */
//  public static  List<String> originalMultirFeatures( String[] tokens, 
//      String[] postags,
//      int[] depParents, String[] depTypes,
//      int[] arg1Pos, int[] arg2Pos, String arg1ner, String arg2ner) {
//
//    List<String> features = new ArrayList<String>();
//
//    // it's easier to deal with first, second
//    int[] first = arg1Pos, second = arg2Pos;
//    String firstNer = arg1ner, secondNer = arg2ner;
//    if (arg1Pos[0] > arg2Pos[0]) {
//      second = arg1Pos; first = arg2Pos;
//      firstNer = arg2ner; secondNer = arg1ner;
//    }
//    
//    // define the inverse prefix
//    String inv = (arg1Pos[0] > arg2Pos[0])? 
//        "inverse_true" : "inverse_false";
//    
//    // define the middle parts
//    StringBuilder middleTokens = new StringBuilder();
//    for (int i=first[1]; i < second[0]; i++) {
//      if (i > first[1]) {
//        middleTokens.append(" ");
//      }
//      middleTokens.append(tokens[i]);
//    }
//    
//    if (second[0] - first[1] > 10) {
//      middleTokens.setLength(0);
//      middleTokens.append("*LONG*");
//    }
//    
//    // define the prefixes and suffixes
//    String[] prefixTokens = new String[2];
//    String[] suffixTokens = new String[2];
//    
//    for (int i=0; i < 2; i++) {
//      int tokIndex = first[0] - i - 1;
//      if (tokIndex < 0) prefixTokens[i] = "B_" + tokIndex;
//      else prefixTokens[i] = tokens[tokIndex];
//    }
//
//    for (int i=0; i < 2; i++) {
//      int tokIndex = second[1] + i;
//      if (tokIndex >= tokens.length) suffixTokens[i] = "B_" + (tokIndex - tokens.length + 1);
//      else suffixTokens[i] = tokens[tokIndex];
//    }
//
//    String[] prefixes = new String[3];
//    String[] suffixes = new String[3];
//
//    prefixes[0] = suffixes[0] = "";
//    prefixes[1] = prefixTokens[0];
//    prefixes[2] = prefixTokens[1] + " " + prefixTokens[0];
//    suffixes[1] = suffixTokens[0];
//    suffixes[2] = suffixTokens[0] + " " + suffixTokens[1];
//    
//    // generate the features in the same order as in ecml data
//    String mto = middleTokens.toString();
//    
//    features.add(inv + "|" + firstNer + "|" + mto + "|" + secondNer);
//    features.add(inv + "|" + prefixes[1] + "|" + firstNer + "|" + mto + "|" + secondNer + "|" + suffixes[1]);
//    features.add(inv + "|" + prefixes[2] + "|" + firstNer + "|" + mto + "|" + secondNer + "|" + suffixes[2]);
//    
//    
//    
//    
//    
//    
//    // dependency features
//    if (depParents == null || depParents.length < tokens.length) return features;
//    
//    // identify head words of arg1 and arg2
//    // (start at end, while inside entity, jump)
//    int head1 = arg1Pos[1]-1;
//    int loopIterationCount =0;
//    while (depParents[head1] >= arg1Pos[0] && depParents[head1] < arg1Pos[1]) {
//        head1 = depParents[head1];
//        //avoid infinite loop
//        if(loopIterationCount == 100){
//          break;
//        }
//        loopIterationCount++;
//    }
//    int head2 = arg2Pos[1]-1;
//    //System.out.println(head1 + " " + head2);
//    loopIterationCount =0;
//    while (depParents[head2] >= arg2Pos[0] && depParents[head2] < arg2Pos[1]) {
//      head2 = depParents[head2];
//      //avoid infinite loop
//        if(loopIterationCount == 100){
//          break;
//        }
//        loopIterationCount++;
//    }
//
//    
//    
//    // find path of dependencies from first to second
//    int[] path1 = new int[tokens.length];
//    for (int i=0; i < path1.length; i++) path1[i] = -1;
//    path1[0] = head1; // last token of first argument
//    for (int i=1; i < path1.length; i++) {
//      path1[i] = depParents[path1[i-1]];
//      if (path1[i] == -1) break;
//    } 
//    int[] path2 = new int[tokens.length];
//    for (int i=0; i < path2.length; i++) path2[i] = -1;
//    path2[0] = head2; // last token of first argument
//    for (int i=1; i < path2.length; i++) {
//      path2[i] = depParents[path2[i-1]];
//      if (path2[i] == -1) break;
//    }
//    int lca = -1;
//    int lcaUp = 0, lcaDown = 0;
//    outer:
//    for (int i=0; i < path1.length; i++)
//      for (int j=0; j < path2.length; j++) {
//        if (path1[i] == -1 || path2[j] == -1) {
//          break; // no path
//        }
//        if (path1[i] == path2[j]) {
//          lca = path1[i];
//          lcaUp = i;
//          lcaDown = j;
//          break outer;
//        }
//      }
//    
//    if (lca < 0) return features; // no dependency path (shouldn't happen)
//    
//    
//    
//    
//    
//    
//    
//    
//    
//    String[] dirs = new String[lcaUp + lcaDown];
//    String[] strs = new String[lcaUp + lcaDown];
//    String[] rels = new String[lcaUp + lcaDown];
//
//    StringBuilder middleDirs = new StringBuilder();
//    StringBuilder middleRels = new StringBuilder();
//    StringBuilder middleStrs = new StringBuilder();
//
//    if (lcaUp + lcaDown < 12) {
//      
//      for (int i=0; i < lcaUp; i++) {
//        dirs[i] = "->";
//        strs[i] = i > 0? tokens[path1[i]] : "";
//        rels[i] = depTypes[path1[i]];
//        //System.out.println("[" + depTypes[path1[i]] + "]->");
//      }
//      for (int j=0; j < lcaDown; j++) {
//      //for (int j=lcaDown-1; j >= 0; j--) {
//        dirs[lcaUp + j] = "<-";
//        strs[lcaUp + j] = (lcaUp + j > 0)? tokens[path2[lcaDown-j]] : ""; // word taken from above
//        rels[lcaUp + j] = depTypes[path2[lcaDown-j]];
//        //System.out.println("[" + depTypes[path2[j]] + "]<-");
//      }
//      
//      for (int i=0; i < dirs.length; i++) {
//        middleDirs.append(dirs[i]);
//        middleRels.append("[" + rels[i] + "]" + dirs[i]);
//        middleStrs.append(strs[i] + "[" + rels[i] + "]" + dirs[i]);
//      }
//    }
//    else {
//        middleDirs.append("*LONG-PATH*");
//        middleRels.append("*LONG-PATH*");
//        middleStrs.append("*LONG-PATH*");
//    }
//  
//    String basicDir = arg1ner + "|" + middleDirs.toString() + "|" + arg2ner;
//    String basicDep = arg1ner + "|" + middleRels.toString() + "|" + arg2ner;
//    String basicStr = arg1ner + "|" + middleStrs.toString() + "|" + arg2ner;
//    
//
//    
//    
//    
//    
//    
//    
//    
//    
//    
//    // new left and right windows: all elements pointing to first arg, but not on path
//    //List<Integer> lws = new ArrayList<Integer>();
//    //List<Integer> rws = new ArrayList<Integer>();
//    
//    List<String> arg1dirs = new ArrayList<String>();
//    List<String> arg1deps = new ArrayList<String>();
//    List<String> arg1strs = new ArrayList<String>();
//    List<String> arg2dirs = new ArrayList<String>();
//    List<String> arg2deps = new ArrayList<String>();
//    List<String> arg2strs = new ArrayList<String>();
//    
//    // pointing out of argument
//    for (int i=0; i < tokens.length; i++) {
//      // make sure itself is not either argument
//      //if (i >= first[0] && i < first[1]) continue;
//      //if (i >= second[0] && i < second[1]) continue;
//      if (i == head1) continue;
//      if (i == head2) continue;
//      
//      // make sure i is not on path
//      boolean onPath = false;
//      for (int j=0; j < lcaUp; j++) if (path1[j] == i) onPath = true;
//      for (int j=0; j < lcaDown; j++) if (path2[j] == i) onPath = true;
//      if (onPath) continue;
//      // make sure i points to first or second arg
//      //if (depParents[i] >= first[0] && depParents[i] < first[1]) lws.add(i);
//      //if (depParents[i] >= second[0] && depParents[i] < second[1]) rws.add(i);
//      if (depParents[i] == head1) {
//        //lws.add(i);
//        arg1dirs.add("->");       
//        arg1deps.add("[" + depTypes[i] + "]->");
//        arg1strs.add(tokens[i] + "[" + depTypes[i] + "]->");
//      }
//      if (depParents[i] == head2) {
//        //rws.add(i);     
//        arg2dirs.add("->");       
//        arg2deps.add("[" + depTypes[i] + "]->");
//        arg2strs.add("[" + depTypes[i] + "]->" + tokens[i]);
//      }
//    }
//    
//    
//    // case 1: pointing into the argument pair structure (always attach to lhs):
//    // pointing from arguments
//    if (lcaUp == 0 && depParents[head1] != -1 || depParents[head1] == head2) {
//      arg1dirs.add("<-");       
//      arg1deps.add("[" + depTypes[head1] + "]<-");
//      arg1strs.add(tokens[head1] + "[" + depTypes[head1] + "]<-");
//      
//      if (depParents[depParents[head1]] != -1) {
//        arg1dirs.add("<-");
//        arg1deps.add("[" + depTypes[depParents[head1]] + "]<-");
//        arg1strs.add(tokens[depParents[head1]] + "[" + depTypes[depParents[head1]] + "]<-");
//      }
//    }
//    // if parent is not on path or if parent is 
//    if (lcaDown == 0 && depParents[head2] != -1 || depParents[head2] == head1) { // should this actually attach to rhs???
//      arg1dirs.add("<-");
//      arg1deps.add("[" + depTypes[head2] + "]<-");
//      arg1strs.add(tokens[head2] + "[" + depTypes[head2] + "]<-");
//      
//      if (depParents[depParents[head2]] != -1) {
//        arg1dirs.add("<-");
//        arg1deps.add("[" + depTypes[depParents[head2]] + "]<-");
//        arg1strs.add(tokens[depParents[head2]] + "[" + depTypes[depParents[head2]] + "]<-");
//      }
//    }
//    
//    // case 2: pointing out of argument
//    
//    //features.add("dir:" + basicDir);    
//    //features.add("dep:" + basicDep);
//
//    
//    // left and right, including word
//    for (String w1 : arg1strs)
//      for (String w2 : arg2strs)
//        features.add("str:" + w1 + "|" + basicStr + "|" + w2);
//    
//    /*
//    for (int lw : lws) {
//      for (int rw : rws) {
//        features.add("str:" + tokens[lw] + "[" + depTypes[lw] + "]<-" + "|" + basicStr
//            + "|" + "[" + depTypes[rw] + "]->" + tokens[rw]);
//      }
//    }
//    */
//    
//    
//    
//    // only left
//    for (int i=0; i < arg1dirs.size(); i++) {
//      features.add("str:" + arg1strs.get(i) + "|" + basicStr);
//      features.add("dep:" + arg1deps.get(i) + "|" + basicDep);
//      features.add("dir:" + arg1dirs.get(i) + "|" + basicDir);
//    }
//    
//    
//    // only right
//    for (int i=0; i < arg2dirs.size(); i++) {
//      features.add("str:" + basicStr + "|" + arg2strs.get(i));
//      features.add("dep:" + basicDep + "|" + arg2deps.get(i));
//      features.add("dir:" + basicDir + "|" + arg2dirs.get(i));
//    }
//
//    features.add("str:" + basicStr);
//
//    return features;
//  }


}
