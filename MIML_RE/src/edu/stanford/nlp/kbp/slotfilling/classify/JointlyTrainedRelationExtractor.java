package edu.stanford.nlp.kbp.slotfilling.classify;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.PrintWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.HashSet;
import java.util.Map;
import java.util.HashMap;

import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Triple;

public abstract class JointlyTrainedRelationExtractor extends RelationExtractor {
  private static final long serialVersionUID = 1L;

  public abstract void train(MultiLabelDataset<String, String> datums);
  
  public static Map<String, Integer[]> score(
      List<Set<String>> goldLabels,
      List<Counter<String>> predictedLabels) {
    // relation -> (truePositive, falseNegative, falsePositive, trueNegative)
    Set<String> rels = new HashSet<String>();
    for (Set<String> goldLabel: goldLabels) {
      rels.addAll(goldLabel);
    }
    Map<String, Integer[]> counter = new HashMap<String, Integer[]>();
    for (String rel: rels) {
      counter.put(rel, new Integer[]{0, 0, 0, 0});
    }

    System.out.println("gold label size: " + goldLabels.size());

    for (int i = 0; i < goldLabels.size(); i++) {
      Set<String> gold = goldLabels.get(i);
      Counter<String> preds = predictedLabels.get(i);
      Set<String> predLabels = preds.keySet();
      for (String rel: rels) {
        if (gold.contains(rel) && predLabels.contains(rel)) {
          // true positive
          counter.get(rel)[0] += 1;
        } else if (gold.contains(rel) && !predLabels.contains(rel)) {
          // false negative
          counter.get(rel)[1] += 1;
        } else if (!gold.contains(rel) && predLabels.contains(rel)) {
          // false positive
          counter.get(rel)[2] += 1;
        } else if (!gold.contains(rel) && !predLabels.contains(rel)) {
          // true negative
          counter.get(rel)[3] += 1;
        }
      }
    }

    return counter;
    /*
    int total = 0, predicted = 0, correct = 0;
    for(int i = 0; i < goldLabels.size(); i ++) {
      Set<String> gold = goldLabels.get(i);
      Counter<String> preds = predictedLabels.get(i);
      total += gold.size();
      predicted += preds.size();
      for(String label: preds.keySet()) {
        if(gold.contains(label)) correct ++;
      }
    }
    
    double p = (double) correct / (double) predicted;
    double r = (double) correct / (double) total;
    double f1 = (p != 0 && r != 0 ? 2*p*r/(p+r) : 0);
    return new Triple<Double, Double, Double>(p, r, f1);
    */
  }
  
  public Map<String, Integer[]> test(
      List<List<Collection<String>>> relations,
      List<Set<String>> goldLabels,
      List<Counter<String>> predictedLabels,
      String predictionFile) throws IOException {
    if(predictedLabels == null)
      predictedLabels = new ArrayList<Counter<String>>();
    for(int i = 0; i < relations.size(); i ++) {
      List<Collection<String>> rel = relations.get(i);
      Counter<String> preds = classifyMentions(rel);
      predictedLabels.add(preds);
    }
    writePredictionFile(goldLabels, predictedLabels, predictionFile);
    return score(goldLabels, predictedLabels);
  }

  public void writePredictionFile(List<Set<String>> goldLabels, List<Counter<String>> predictedLabels, String predictionFile) throws IOException {
    PrintWriter writer = new PrintWriter(new FileWriter(predictionFile, true));
    
    for (int i = 0; i < goldLabels.size(); i++) {
      Set<String> trueLabel = goldLabels.get(i);
      Set<String> predLabel = predictedLabels.get(i).keySet();

      writer.println(helper(trueLabel) + "\t" + helper(predLabel));
    }

    writer.close();
  }

  public String helper(Set<String> labels) {
    if (labels.size() == 0) {
      return "";
    }
    StringBuilder sb = new StringBuilder();
    for (String label: labels) {
      sb.append(label);
      sb.append(", ");
    }
    return sb.substring(0, sb.length()-2);
  }
  
  public Triple<Double, Double, Double> oracle(
      List<List<Collection<String>>> relations,
      List<Set<String>> goldLabels,
      List<Counter<String>> predictedLabels) {
    if(predictedLabels == null)
      predictedLabels = new ArrayList<Counter<String>>();
    for(int i = 0; i < relations.size(); i ++) {
      List<Collection<String>> rel = relations.get(i);
      Counter<String> preds = classifyOracleMentions(rel, goldLabels.get(i));
      predictedLabels.add(preds);
    }
    System.out.println("You want to be careful! We are using this function!");
    return new Triple<Double, Double, Double>(0.0, 0.0, 0.0);
    //return score(goldLabels, predictedLabels);
  }
  
  public abstract void save(String path) throws IOException;
  public abstract void load(ObjectInputStream in) throws IOException, ClassNotFoundException;
}
