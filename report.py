
import time 
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import pandas as pd 
import numpy as np

def generateReport(model, testx, testy, targetNames, training_history, training_time, crop_time, params) -> str:
	""" 
	train time 
	crop time 
	test time per image 
	accuracy 
		accuracy per output
	size 
	summary 
	training history 
	Results per output: #######################
		precision, recall 
		confusion matrix, 
		--------------------
	"""
	# training time 
	results = 'Training time: \t{} minutes\n'.format(training_time) 
	results += '   Crop Time: \t{} milliseconds\n'.format(crop_time)
	
	# test time per image 
	starttest = time.time()
	(apredy, gpredy, epredy) = model.predict(testx)
	results += '   Test Time: \t{} milliseconds\n'.format(1000 * float(time.time() - starttest) / testx.shape[0])
	
	# TEMP: calculate metrics
	results += 'Metrics:\n'
	acc = 1.0
	eval = model.evaluate(testx, testy)
	for i, value in enumerate(eval):
		results += '  - {} \t:  {}\n'.format(model.metrics_names[i],value)
		if 'accuracy' in model.metrics_names[i]:
			acc *= float(value)
	results += '\n'
	
	# Parameters
	results += 'Parameters:\n'
	for key, value in params.items():
		results += '  - {} \t:  {}\n'.format(key,value)
			
	results += '\n'
	
	# size of model 
	model_save_file = 'testmodel.keras' 
	model.save (model_save_file) # saveModel(model)
	size = Path(model_save_file).stat().st_size
	results += '  Model Size: \t{} GB \n\n'.format(size/1073741824)
	
	# Model Summary 
	stringlist = []
	model.summary(print_fn=lambda x: stringlist.append(x))
	short_model_summary = "\n".join(stringlist)
	results += short_model_summary + '\n\n'
	
	# Training Summary 
	results += pd.DataFrame.from_dict(training_history.history,orient='index').transpose().to_string() + '\n\n'
	
	# Title 
	barrier = '# ' + '=' * 78 + '\n'
	results += barrier + '# Results\n' + barrier +'\n'
	
	for i, (output, y_pred) in enumerate([('age', apredy), ('gender', gpredy), ('ethnicity', epredy)]):
		# 1 - sum 
		# re-add column after processing model
		new_col = np.ones(testy[output].shape[0]) - testy[output].sum(axis=1)
		new_col = new_col.reshape(new_col.shape[0], 1)
		y_test = np.append(testy[output], new_col, 1)
		
		# re-add column after processing model
		new_col = np.ones(y_pred.shape[0]) - y_pred.sum(axis=1)
		new_col = new_col.reshape(new_col.shape[0], 1)
		y_pred = np.append(y_pred, new_col, 1)
		
		y_test = y_test.argmax(axis=1)
		y_pred = y_pred.argmax(axis=1)
		
		# Precision and Recall
		results += classification_report(y_test, y_pred, labels=list(range(len(targetNames[output]))), target_names=targetNames[output])
		results += '\n\n'
		
		# confusion matrix
		matrix = confusion_matrix(y_test, y_pred, labels = list(range(len(targetNames[output]))))
		matrixSTR = '\n'.join('\t'.join('%0.3f' %x for x in y) for y in matrix)
		results += 'Confusion Matrix-{}-\n{}\n\n'.format(i, matrixSTR)
		results += '\n' + '-'*40 + '\n\n' # add separator
		
	return results, acc