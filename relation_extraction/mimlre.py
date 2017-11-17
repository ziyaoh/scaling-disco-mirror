from modelTest import get_f1_score, get_confusion_table

def evaluate(pred_file, report):
  pred = {}
  test = {}
  f1s = {}
  labels = []
  with open(pred_file, 'r') as preds:
    for i, ins in enumerate(preds):
      if i == 0:
        labels = ins.split(', ')
        for label in labels:
          pred[label] = []
          test[label] = []
          f1s[label] = []
      else:
        test_labels_all, pred_labels_all = ins.split('\t')
        test_labels = test_labels_all.split(', ')
        pred_labels = pred_labels_all.split(', ')

  print labels
  overall_f1 = 0.0
  for label in labels:
      f1s[label] = (get_f1_score(pred[label], test[label], 'macro'), get_confusion_table(pred[label], test[label], (0, 1)))
      overall_f1 += f1s[label][0]
  overall_f1 /= len(labels)

  with open(report, 'w') as writer:
      for label in f1s:
          writer.write("%s\t%s\n" % (label, f1s[label][0]))
          confusion_table = f1s[label][1]
          writer.write("\t%s\t%s\n" % (1, 0))
          writer.write("1\t%s\t%s\n" % (confusion_table[1][1], confusion_table[1][0]))
          writer.write("0\t%s\t%s\n" % (confusion_table[0][1], confusion_table[0][0]))
      writer.write("overall F1: %s\n" % overall_f1)
      writer.write("\n")
