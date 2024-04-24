from heuristics_algorithm import Heuristics_Algorithm

main_dir = '../content/drive/MyDrive/data/s1/'

alg = Heuristics_Algorithm(budget=631064, qPairs = 312327, delimiter='\t',  sourceFilePath=main_dir + 'sourceSample.tsv', targetFilePath=main_dir + 'targetSample.tsv', wScheme = 'CF',  HeuristicCondition = "Precision_Threshold", ConditionLimit = 0.5 , DynamicFactor = 0, ViolationLimit = 0)
alg.applyProcessing()
