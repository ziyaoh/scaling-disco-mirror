import sys, argparse
import dataParse
import classifier


def readCommand(argv):
	"""
	input: input file name
	test: test file name
	-c: classifier type
	-f: feature type
	-o: output file name
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('input', help = 'the input file name')
	parser.add_argument('test', help = 'the held out test data')
	parser.add_argument('-f', '--feature',
						default = 'default',
						help = 'the type of feature we extract from the input data')
	parser.add_argument('-c', '--classifier',
						default = 'logit',
						choices = ['logit', "NN"],
						help = 'the type of classifier we want to use')
	parser.add_argument('-o', '--output',
						default = 'result.out',
						help = 'the output file name')

	return parser.parse_args()

def abnormalDataHandle(abnormalData, filename):
		print "number of abnormal data instances in", filename, ":", len(abnormalData)
		while True:
			action = raw_input("'q' to exit, 'p' to print the abnormal data instances, Enter to continue")
			if action == 'q':
				sys.exit(0)
			elif action == 'p':
				for sentence in abnormalData:
					print sentence
			elif action == '':
				return
			else:
				print "unknown command"

def buildModel(opt):
	'''
	Read training data from input file, fit a classifier model according to the training data.
	Read testing data from test file, and test the classifier model.
	'''
	(data, abnormalData) = dataParse.read_data(opt.input)
	(data_test, abnormalData_test) = dataParse.read_data(opt.test)
	if len(abnormalData) > 0:
		abnormalDataHandle(abnormalData, opt.input)
	if len(abnormalData_test) > 0:
		abnormalDataHandle(abnormalData_test, opt.test)

	X, y = dataParse.format_data(data)
	X_test, y_test = dataParse.format_data(data_test)

	my_classifier = classifier.LinearClassifier(opt.feature, opt.classifier)
	my_classifier.fit(X, y)

	result = my_classifier.predict(X_test)

	good = 0
	for i, pred in enumerate(result):
		if pred == y_test[i]:
			good += 1

	print "correct\t\tall"
	print "%s\t\t%s" %(good, len(result))

if __name__ == '__main__':
	opt = readCommand(sys.argv[1:])
	buildModel(opt)



		