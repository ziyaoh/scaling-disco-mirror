RawTrainingFile=$1
RawOriginalEvalFile=$2
#RawModifiedEvalFile=$3

TrainingFile="featurized_$RawTrainingFile"
OriginalEvalFile="featurized_$RawOriginalEvalFile"
#ModifiedEvalFile="featurized_$RawModifiedEvalFile"

cd featureGeneration/MultirFramework
sbt clean compile
sbt "runMain edu.washington.multirframework.featuregeneration.DefaultFeatureGenerator ../../$RawTrainingFile ../../$TrainingFile"
sbt "runMain edu.washington.multirframework.featuregeneration.DefaultFeatureGenerator ../../$RawOriginalEvalFile ../../$OriginalEvalFile"
#sbt "runMain edu.washington.multirframework.featuregeneration.DefaultFeatureGenerator ../../$RawModifiedEvalFile ../../$ModifiedEvalFile"
cd ../../

cd relation_extraction
python relationExtraction.py ../$TrainingFile ../$OriginalEvalFile
#cat "${TrainingFile}.score"
