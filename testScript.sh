cd featureGeneration/MultirFramework
sbt clean compile
sbt "runMain edu.washington.multirframework.featuregeneration.DefaultFeatureGenerator ../../testTrain_input.txt ../../testTrain.txt"
sbt "runMain edu.washington.multirframework.featuregeneration.DefaultFeatureGenerator ../../testEval_input.txt ../../testEval.txt"

cd ../../relation_extraction
python relationExtraction.py ../testTrain.txt ../testEval.txt
cat testTrain.score
