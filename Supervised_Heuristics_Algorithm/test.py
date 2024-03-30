from heuristics_algorithm import Heuristics_Algorithm

main_dir = '../content/drive/MyDrive/data/s1/'

sg = Heuristics_Algorithm(budget=5000000, qPairs = 100, delimiter='\t',  sourceFilePath=main_dir + 'sourceSample.tsv', targetFilePath=main_dir + 'targetSample.tsv', HeuristicCondition = "Dynamic_Qualifying_Distance_Threshold", ConditionLimit = 10 , DynamicFactor = 0.5, ViolationLimit = 0)
sg.applyProcessing()
