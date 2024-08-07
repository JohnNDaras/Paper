from heuristics_algorithm import Heuristics_Algorithm

main_dir = '../content/drive/MyDrive/s9/'
sg = Heuristics_Algorithm(budget=5679576, qPairs = 2362497, delimiter='\t',  sourceFilePath=main_dir + 'sourceSample.tsv', targetFilePath=main_dir + 'targetSample.tsv', HeuristicCondition = "Dynamic_Qualifying_Distance_Threshold", ConditionLimit = 10000 , DynamicFactor = 0.5, ViolationLimit = 0)
sg.applyProcessing()
