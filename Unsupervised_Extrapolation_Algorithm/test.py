from extrapolation import Extrapolation

main_dir = '../content/drive/MyDrive/data/s1/'
x=0
print('Enter desired recall:')
x = input()
alg = Extrapolation(budget=500000, qPairs = 100, delimiter='\t',  sourceFilePath=main_dir + 'sourceSample.tsv', targetFilePath=main_dir + 'targetSample.tsv', wScheme = 'CF',  users_input=float(x))
alg.applyProcessing()
