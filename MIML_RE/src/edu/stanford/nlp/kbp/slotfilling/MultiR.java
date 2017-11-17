package edu.stanford.nlp.kbp.slotfilling;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.zip.GZIPInputStream;
import java.lang.StringBuilder;
import java.lang.reflect.*;

import edu.stanford.nlp.kbp.slotfilling.classify.*;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.ProcessWrapper;
import edu.stanford.nlp.kbp.slotfilling.common.Props;
import edu.stanford.nlp.kbp.slotfilling.multir.ProtobufToMultiLabelDataset;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.PropertiesUtils;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.Triple;
import edu.stanford.nlp.util.ErasureUtils;
import edu.stanford.nlp.util.HashIndex;
import edu.stanford.nlp.util.Index;

/**
 * Trains and evaluates on the MultiR corpus (Hoffmann et al., 2011)
 * @author Mihai
 *
 */
public class MultiR {
  static final int TUNING_FOLDS = 3;
  
  static class Parameters {
    static final String DEFAULT_TYPE = "jointbayes";
    static final int DEFAULT_FEATURE_COUNT_THRESHOLD = 5;
    static final int DEFAULT_EPOCHS = 15;
    static final int DEFAULT_FOLDS = 5;
    static final String DEFAULT_FILTER = "all";
    static final String DEFAULT_INF_TYPE = "stable";
    static final int DEFAULT_MODEL = 0;
    static final boolean DEFAULT_TRAINY = true;
    
    static final String FOLD_PROP = "fold";

    String initializeFile;
    String trainFile;
    String testFile;
    ModelType type;
    int featureCountThreshold;
    int numberOfTrainEpochs;
    int numberOfFolds;
    String workDir;
    String baseDir;
    String localFilter;
    int featureModel;
    String infType;
    boolean trainY;
    Integer fold;
    
    static Parameters propsToParameters(Properties props) {
      Parameters p = new Parameters();
      p.initializeFile = props.getProperty("multir.initialize");
      p.trainFile = props.getProperty("multir.train");
      p.testFile = props.getProperty("multir.test");
      p.type = ModelType.stringToModel(props.getProperty(
          Props.MODEL_TYPE,  
          DEFAULT_TYPE));
      p.featureCountThreshold = PropertiesUtils.getInt(props, 
          Props.FEATURE_COUNT_THRESHOLD, 
          DEFAULT_FEATURE_COUNT_THRESHOLD);
      p.numberOfTrainEpochs = PropertiesUtils.getInt(props, 
          Props.EPOCHS, 
          DEFAULT_EPOCHS);
      p.numberOfFolds = PropertiesUtils.getInt(props, 
          Props.FOLDS, 
          DEFAULT_FOLDS);
      p.localFilter = props.getProperty(
          Props.FILTER,
          DEFAULT_FILTER);
      p.featureModel = PropertiesUtils.getInt(props, 
          Props.FEATURES,
          DEFAULT_MODEL);
      p.infType = props.getProperty(
          Props.INFERENCE_TYPE,
          DEFAULT_INF_TYPE);
      p.trainY = PropertiesUtils.getBool(props,
          Props.TRAINY,
          DEFAULT_TRAINY);
      p.workDir = props.getProperty(Props.WORK_DIR);
      p.baseDir = props.getProperty(Props.CORPUS_BASE_DIR);
      p.fold = props.containsKey(FOLD_PROP) ?
          Integer.valueOf(props.getProperty(FOLD_PROP)) : null;
      return p;
    }
  }
  
  public static void main(String[] args) throws Exception {
    Properties props = StringUtils.argsToProperties(args);
    Log.setLevel(Log.stringToLevel(props.getProperty(Props.LOG_LEVEL, "SEVERE")));

    Log.severe("running: multiR");
    // Log.severe("training file: " + args[0]);

    Parameters p = Parameters.propsToParameters(props);

    //String[] initialFiles = {"none", "corpora/multir/CS_9000", "corpora/multir/CS_18000"};
    //String[] trainFiles = {"corpora/multir/DS_all", "corpora/multir/DS_all_CS_9000", "corpora/multir/DS_all_CS_18000"};
    
    String[] initialFiles = {"none", "corpora/multir/CS_9000_forNeg", "corpora/multir/CS_18000_forNeg"};
    String[] trainFiles = {"corpora/multir/DS_all_neg_1", "corpora/multir/DS_all_CS_9000_neg_1", "corpora/multir/DS_all_CS_18000_neg_1"};
    
    //String[] initialFiles = {"corpora/multir/CS_9000"};
    //String[] trainFiles = {"corpora/multir/DS_1000_CS_9000"};
    
    // straight run, train and test
    for (int i = 0; i < initialFiles.length; i++) {
      p.initializeFile = initialFiles[i];
      p.trainFile = trainFiles[i];
      run(p);
    }
  
  }
  
  private static void run(Parameters p) throws Exception {
    /*
    initialize model with random data, save initial, do nothing else
    train the initial model with DS data
    test with test data, generate report
    */

    // spectating all fields in parameter p
    // for (Field field : p.getClass().getDeclaredFields()) {
    //   field.setAccessible(true);
    //   String name = field.getName();
    //   Object value = field.get(p);
    //   System.out.printf("Field name: %s, Field value: %s%n", name, value);
    // }

    Log.severe("initial file: " + (p.initializeFile.equals("none")));
    String sig = makeSignature(p);
    Log.severe("Using signature: " + sig);
    String modelPath = p.workDir + File.separator + sig + ".ser";
    String scoreFile = p.workDir + File.separator + sig + ".score";
    String predictionFile = p.workDir + File.separator + sig + ".pred";
    // boolean showPRCurve = PropertiesUtils.getBool(props, Props.SHOW_CURVE, true);

    Index<String> labelIndex = new HashIndex<String>();
    Index<String> featureIndex = new HashIndex<String>();

/*
    MultiLabelDataset<String, String> trainDataset = parseData(p.trainFile, featureIndex, labelIndex, p.featureCountThreshold);
    if (!p.initializeFile.equals("none")) {
      // initialize
      Log.severe("initializing with " + p.initializeFile);
      MultiLabelDataset<String, String> initializeDataset = parseData(p.initializeFile, featureIndex, labelIndex, p.featureCountThreshold);
      run(p, modelPath, initializeDataset, false);
    }
*/

    List<MultiLabelDataset<String, String>> datasets = parseData(p.initializeFile, p.trainFile, featureIndex, labelIndex, p.featureCountThreshold);
    MultiLabelDataset<String, String> trainDataset = datasets.get(1);
    MultiLabelDataset<String, String> initialDataset = datasets.get(0);

    // write relations to pred file
    List<String> labels = labelIndex.objectsList();
    StringBuilder sb = new StringBuilder();
    for (String label: labels) {
      sb.append(label);
      sb.append(", ");
    }
    String labels_output = sb.substring(0, sb.length() - 2);

    PrintWriter writer = new PrintWriter(predictionFile, "UTF-8");
    writer.println(labels_output);
    writer.close();

    if (initialDataset != null) {
      Log.severe("initializing with " + p.initializeFile);
      run(p, modelPath, initialDataset, false);
    }

    // train with DS
    Log.severe("training with " + p.trainFile);
    JointlyTrainedRelationExtractor extractor = run(p, modelPath, trainDataset, true);

    // test
    Log.severe("testing with " + p.testFile);
    List<Set<String>> goldLabels = new ArrayList<Set<String>>();
    List<Counter<String>> predictedLabels = new ArrayList<Counter<String>>();
    List<List<Collection<String>>> relations = 
      new ArrayList<List<Collection<String>>>();
    BufferedReader bf = new BufferedReader(new FileReader(p.testFile));
    parseTestData(bf, relations, goldLabels);
    bf.close();
    Map<String, Integer[]> counter = extractor.test(relations, goldLabels, predictedLabels, predictionFile);
    //Triple<Double, Double, Double> score = extractor.test(relations, goldLabels, predictedLabels);
    //Triple<Double, Double, Double> score = extractor.oracle(relations, goldLabels, predictedLabels);

    Map<String, Triple<Double, Double, Double>> stats = new HashMap<String, Triple<Double, Double, Double>>();
    Set<String> rels = counter.keySet();
    for (String rel: rels) {
      Integer[] table = counter.get(rel);
      int TP = table[0];
      int FP = table[2];
      int FN = table[1];

      double precision = (TP == 0)? 0: (double)TP / (double)(TP + FP);
      double recall = (TP == 0)? 0: (double)TP / (double)(TP + FN);
      double f1 = (precision != 0 && recall != 0)? 2 * precision * recall / (precision + recall): 0;

      stats.put(rel, new Triple(precision, recall, f1));
    }
    
    String output = getResultOutput(stats, counter);
    System.out.println(output);
    PrintStream os = new PrintStream(new FileOutputStream(scoreFile));
    os.println(output);
    os.close();

    /* 
    System.out.println("P " + score.first() + " R " + score.second() + " F1 " + score.third());
    
    PrintStream os = new PrintStream(new FileOutputStream(scoreFile));
    os.println("P " + score.first() + " R " + score.second() + " F1 " + score.third());
    os.close();
    */
    
    // if(showPRCurve) {
    //   String curveFile = p.workDir + File.separator + sig + ".curve";
    //   os = new PrintStream(new FileOutputStream(curveFile));
    //   // generatePRCurve(os, goldLabels, predictedLabels);
    //   generatePRCurveNonProbScores(os, goldLabels, predictedLabels);
    //   os.close();
    //   System.out.println("P/R curve values saved in file " + curveFile);
    // }
  }


  private static JointlyTrainedRelationExtractor run(
      Parameters p, 
      String modelPath,
      MultiLabelDataset<String, String> trainDataset,
      boolean trainWithDS) throws IOException, ClassNotFoundException {
    JointlyTrainedRelationExtractor extractor = null;
    String initialModelPath = modelPath.replaceAll("\\.ser", ".initial.ser");
    if(trainWithDS) {
      JointBayesRelationExtractor ex = new JointBayesRelationExtractor(
          initialModelPath, 
          p.numberOfTrainEpochs, 
          p.numberOfFolds, 
          p.localFilter, 
          p.featureModel,
          p.infType,
          p.trainY,
          false);
      ex.setSerializedModelPath(modelPath);
      extractor = ex;
    } else {
      extractor = new JointBayesRelationExtractor(
          initialModelPath, 
          p.numberOfTrainEpochs, 
          p.numberOfFolds, 
          p.localFilter, 
          p.featureModel,
          p.infType,
          p.trainY,
          true);
    }
    
    if(new File(modelPath).exists()) {
      // load an existing model
      Log.severe("Existing model found at " + modelPath + ". Will NOT train a new one.");
      ObjectInputStream in = new ObjectInputStream(new FileInputStream(modelPath));
      extractor.load(in);
      in.close();
    } else {
      
      extractor.train(trainDataset);
      if (trainWithDS) {
        // save
        extractor.save(modelPath);
      }
    }
    
    return extractor;
  }

  private static String getResultOutput(Map<String, Triple<Double, Double, Double>> stats, Map<String, Integer[]> counter) {
    StringBuilder output = new StringBuilder();
    double totalF1 = 0.0;
    for (String rel:stats.keySet()) {
      output.append(rel + ":\n");
      Integer[] confTable = counter.get(rel);
      output.append("\ttrue\tfalse\n");
      output.append(String.format("true\t%d\t%d\n", confTable[0], confTable[1]));
      output.append(String.format("false\t%d\t%d\n", confTable[2], confTable[3]));
      Triple<Double, Double, Double> stat = stats.get(rel);
      String oneRelOutput = String.format("precision: %f\nrecall: %f\nf1: %f\n\n", stat.first(), stat.second(), stat.third());
      output.append(oneRelOutput);
      totalF1 += stat.third();
    }
    output.append("overall F1: " + totalF1/stats.size());
    return output.toString();
  }

  private static List<MultiLabelDataset<String, String>> parseData(
    String initialDataFile, 
    String trainDataFile,
    Index<String> featureIndex, 
    Index<String> labelIndex,
    int featureCountThreshold) throws IOException {

    Collection<EntityObject> objects_train = new HashSet<EntityObject>();
    int[][][] data_train = makeData(trainDataFile, featureIndex, objects_train);
    Collection<EntityObject> objects_initial = new HashSet<EntityObject>();
    int[][][] data_initial = null;
    if (!initialDataFile.equals("none")) {
      data_initial = makeData(initialDataFile, featureIndex, objects_initial);
    }

    float[] counts = getFeatureCounts(featureIndex, data_train);
    int[] featMap = new int[featureIndex.size()];
    featureIndex = applyFeatureCountThresholdOnFeatureIndex(featureIndex, counts, featureCountThreshold, featMap);
    applyFeatureCountThresholdOnData(featureIndex, data_train, featMap);
    if (data_initial != null) {
      applyFeatureCountThresholdOnData(featureIndex, data_initial, featMap);
    }

    MultiLabelDataset<String, String> trainDataset = toDataset(data_train, objects_train, featureIndex, labelIndex);
    MultiLabelDataset<String, String> initialDataset = initialDataFile.equals("none")? null: toDataset(data_initial, objects_initial, featureIndex, labelIndex);

    trainDataset.randomize(1);
    if (initialDataset != null) {
      initialDataset.randomize(1);
    }
//    trainDataset.applyFeatureCountThreshold(featureCountThreshold);
//    MultiLabelDataset<String, String>[] result = {initialDataset, trainDataset};
    List<MultiLabelDataset<String, String>> result = new ArrayList<MultiLabelDataset<String, String>>();
    result.add(initialDataset);
    result.add(trainDataset);
    return result;
  }

  private static int[][][] makeData(String dataFile, Index<String> featureIndex, Collection<EntityObject> objects) throws IOException {
    BufferedReader bf = new BufferedReader(new FileReader(dataFile));
    objects.addAll(toEntityObjects(bf, true));
    int[][][] data = toData(objects, featureIndex);
    bf.close();
    return data;
  }

  private static int[][][] toData(Collection<EntityObject> objects, Index<String> featureIndex) {
    int offset = 0;
    int[][][] data = new int[objects.size()][][];
    for (EntityObject obj: objects) {
      int [][] group = new int[obj.mentions.size()][];

      for (int i = 0; i < obj.mentions.size(); i++) {
        List<String> sfeats = obj.mentions.get(i);
        int[] features = new int[sfeats.size()];
        for (int j = 0; j < sfeats.size(); j++) {
          int featureInd = featureIndex.indexOf(sfeats.get(j), true);
          if (featureInd >= featureIndex.size()) {
            Log.severe("Feature index too large! Total number: " + featureIndex.size() + ", index got: " + featureInd);
          }
          features[j] = featureInd;
        }
        group[i] = features;
      }

      data[offset] = group;

      offset ++;
    }
    return data;
  }

  private static Index<String> applyFeatureCountThresholdOnFeatureIndex(Index<String> featureIndex, float[] counts, int threshold, int[] featMap) {
    Index<String> newFeatureIndex = new HashIndex<String>();
    for (int i = 0; i < featMap.length; i++) {
      String feat = featureIndex.get(i);
      if (counts[i] >= threshold) {
        int newIndex = newFeatureIndex.size();
        newFeatureIndex.add(feat);
        featMap[i] = newIndex;
      } else {
        featMap[i] = -1;
      }
    }

    return newFeatureIndex;
  }

  private static void applyFeatureCountThresholdOnData(Index<String> featureIndex, int[][][] data, int[] featMap) {
    for (int i = 0; i < data.length; i++) {
      for(int j = 0; j < data[i].length; j ++){
        List<Integer> featList = new ArrayList<Integer>(data[i][j].length);
        for (int k = 0; k < data[i][j].length; k++) {
          if (featMap[data[i][j][k]] >= 0) {
            featList.add(featMap[data[i][j][k]]);
          }
        }
        data[i][j] = new int[featList.size()];
        for(int k = 0; k < data[i][j].length; k ++) {
          data[i][j][k] = featList.get(k);
        }
      }
    }
  }

  private static float[] getFeatureCounts(Index<String> featureIndex, int[][][] data) {
    float[] counts = new float[featureIndex.size()];
    for (int i = 0; i < data.length; i++) {
      for (int j = 0; j < data[i].length; j++) {
        for(int k = 0; k < data[i][j].length; k ++) {
          counts[data[i][j][k]] += 1.0;
        }
      }
    }
    return counts;
  }

  private static MultiLabelDataset<String, String> toDataset(
    int[][][] data,
    Collection<EntityObject> objects,
    Index<String> featureIndex,
    Index<String> labelIndex) throws IOException {
    /*
    1. parse each input instance to a EntityObject object:
    2. parse this list of map into MultiLabelDataset
    */

    // featureIndex = (featureIndex == null)? new HashIndex<String>(): featureIndex;
    // labelIndex = (labelIndex == null)? new HashIndex<String>(): labelIndex;
    Set<Integer> [] posLabels = ErasureUtils.<Set<Integer> []>uncheckedCast(new Set[objects.size()]);
    Set<Integer> [] negLabels = ErasureUtils.<Set<Integer> []>uncheckedCast(new Set[objects.size()]);

    int offset = 0;
    for (EntityObject obj: objects) {
      Set<Integer> pos = new HashSet<Integer>();
      Set<Integer> neg = new HashSet<Integer>();
      for(String l: obj.posLabels) {
        pos.add(labelIndex.indexOf(l, true));
      }
      for(String l: obj.negLabels) {
        neg.add(labelIndex.indexOf(l, true));
      }
      posLabels[offset] = pos;
      negLabels[offset] = neg;
      /*
      int [][] group = new int[obj.mentions.size()][];

      for (int i = 0; i < obj.mentions.size(); i++) {
        List<String> sfeats = obj.mentions.get(i);
        int[] features = new int[sfeats.size()];
        for (int j = 0; j < sfeats.size(); j++) {
          int featureInd = featureIndex.indexOf(sfeats.get(j), true);
          if (featureInd >= featureIndex.size()) {
            Log.severe("Feature index too large! Total number: " + featureIndex.size() + ", index got: " + featureInd);
          }
          features[j] = featureInd;
        }
        group[i] = features;
      }

      data[offset] = group;
      */
      offset ++;
    }

    MultiLabelDataset<String, String> dataset = new MultiLabelDataset<String, String>(
      data, featureIndex, labelIndex, posLabels, negLabels);
    return dataset;
  }

  private static void parseTestData(BufferedReader bf,
      List<List<Collection<String>>> relationFeatures,
      List<Set<String>> labels) throws IOException {
    Collection<EntityObject> objects = toEntityObjects(bf, false);
    // toDatums(relations, relationFeatures, labels);
    for(EntityObject obj: objects) {
      labels.add(obj.posLabels);
      List<Collection<String>> mentionFeatures = new ArrayList<Collection<String>>();
      // instanceFeatures.add(ins.features);
      for(int i = 0; i < obj.mentions.size(); i ++){
        mentionFeatures.add(obj.mentions.get(i));
      }
      relationFeatures.add(mentionFeatures);
    }
    assert(labels.size() == relationFeatures.size());
  }

  private static Collection<EntityObject> toEntityObjects(BufferedReader bf, boolean generateNegativeLabels) throws IOException {
    // List<EntityObject> objects = new ArrayList<EntityObject>();
    Map<String, EntityObject> objects = new HashMap<String, EntityObject>();
    Set<String> allPosRelations = new HashSet<String>();

    String sentence = null;
    while ((sentence = bf.readLine()) != null) {
      // parse sentence to EntityObject with features and posLabels filled
      String[] fields = sentence.split("\\t");
      //parse two entities to get key
      String key = "(" + fields[1] + "," + fields[4] + ")";

      EntityObject obj = objects.get(key);
      if (obj == null) {
        obj = new EntityObject();
        objects.put(key, obj);
      }
      // parse posLabels
      String[] labels = fields[7].replace("[", "").replace("]", "").split(", ");
      for (String label: labels) {
        if (!label.equals("")) {
          obj.posLabels.add(label);
        }
      }
      // add all features
      List<String> features = new ArrayList<String>();
      for (int i = 8; i < fields.length; i++) {
        features.add(fields[i]);
      }
      obj.mentions.add(features);

      allPosRelations.addAll(obj.posLabels);
    }

    // make negatives
    if(generateNegativeLabels) {
      for(EntityObject obj: objects.values()) {
        Set<String> negatives = new HashSet<String>(allPosRelations);
        negatives.removeAll(obj.posLabels);
        obj.negLabels = negatives;
      }
    }
    return objects.values();
  }

  private static class EntityObject {
    Set<String> posLabels;
    Set<String> negLabels;
    List<List<String>> mentions;

    public EntityObject() {
      this.posLabels = new HashSet<String>();
      this.negLabels = new HashSet<String>();
      this.mentions =  new ArrayList<List<String>>();
    }
  }
  
  private static String makeSignature(Parameters p) {
    // StringBuffer os = new StringBuffer();
    // os.append("multir");
    // os.append("_" + p.type);
    // os.append("_T" + p.featureCountThreshold);
    // os.append("_E" + p.numberOfTrainEpochs);
    // os.append("_NF" + p.numberOfFolds);
    // os.append("_F" + p.localFilter);
    // os.append("_M" + p.featureModel);
    // os.append("_I" + p.infType);
    // os.append("_Y" + p.trainY);

    // // in case of cross-validation tuning
    // if(p.fold != null) 
    //   os.append("_fold" + p.fold);
    
    // return os.toString();
    String[] fields = p.trainFile.split("/");
    return fields[fields.length - 1];
  }


  /*
  private static void generatePRCurveNonProbScores(PrintStream os,
                                                   List<Set<String>> goldLabels,
                                                   List<Counter<String>> predictedLabels) {
    // each triple stores: position of tuple in gold, one label for this tuple, its score
    List<Triple<Integer, String, Double>> preds = convertToSorted(predictedLabels);
    double prevP = -1, prevR = -1;
    int START_OFFSET = 10; // score at least this many predictions (makes no sense to score 1...)
    for(int i = START_OFFSET; i < preds.size(); i ++) {
      List<Triple<Integer, String, Double>> filteredLabels = preds.subList(0, i);
      Triple<Double, Double, Double> score = score(filteredLabels, goldLabels);
      if(score.first() != prevP || score.second() != prevR) {
        double ratio = (double) i / (double) preds.size();
        os.println(ratio + " P " + score.first() + " R " + score.second() + " F1 " + score.third());
        prevP = score.first();
        prevR = score.second();
      }
    }
  }
  private static List<Triple<Integer, String, Double>> convertToSorted(List<Counter<String>> predictedLabels) {
    List<Triple<Integer, String, Double>> sorted = new ArrayList<Triple<Integer, String, Double>>();
    for(int i = 0; i < predictedLabels.size(); i ++) {
      for(String l: predictedLabels.get(i).keySet()) {
        double s = predictedLabels.get(i).getCount(l);
        sorted.add(new Triple<Integer, String, Double>(i, l, s));
      }
    }
    Collections.sort(sorted, new Comparator<Triple<Integer, String, Double>>() {
      @Override
      public int compare(Triple<Integer, String, Double> t1, Triple<Integer, String, Double> t2) {
        if(t1.third() > t2.third()) return -1;
        else if(t1.third() < t2.third()) return 1;
        return 0;
      }
    });
    return sorted;
  }
  private static Triple<Double, Double, Double> score(List<Triple<Integer, String, Double>> preds, List<Set<String>> golds) {
    int total = 0, predicted = 0, correct = 0;
    for(int i = 0; i < golds.size(); i ++) {
      Set<String> gold = golds.get(i);
      total += gold.size();
    }
    for(Triple<Integer, String, Double> pred: preds) {
      predicted ++;
      if(golds.get(pred.first()).contains(pred.second()))
        correct ++;
    }

    double p = (double) correct / (double) predicted;
    double r = (double) correct / (double) total;
    double f1 = (p != 0 && r != 0 ? 2*p*r/(p+r) : 0);
    return new Triple<Double, Double, Double>(p, r, f1);
  }

  
  private static void generatePRCurve(PrintStream os,
      List<Set<String>> goldLabels, 
      List<Counter<String>> predictedLabels) {
    for(double t = 1.0; t >= 0; ) {
      List<Counter<String>> filteredLabels = keepAboveThreshold(predictedLabels, t);
      Triple<Double, Double, Double> score = JointlyTrainedRelationExtractor.score(goldLabels, filteredLabels);
      os.println(t + " P " + score.first() + " R " + score.second() + " F1 " + score.third());
      if(t > 1.0) t -= 1.0;
      else if(t > 0.99) t -= 0.0001;
      else if(t > 0.95) t -= 0.001;
      else t -= 0.01;
    }
  }
  
  private static List<Counter<String>> keepAboveThreshold(List<Counter<String>> labels, double threshold) {
    List<Counter<String>> filtered = new ArrayList<Counter<String>>();
    for(Counter<String> group: labels) {
      Counter<String> filteredGroup = new ClassicCounter<String>();
      for(String l: group.keySet()) {
        double v = group.getCount(l);
        if(v >= threshold) filteredGroup.setCount(l, v);
      }
      filtered.add(filteredGroup);
    }
    return filtered;
  }
  */
}
