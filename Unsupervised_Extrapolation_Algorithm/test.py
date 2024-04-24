from extrapolation import Extrapolation

main_dir = '../content/drive/MyDrive/s9/'
x=0
print('Enter desired recall:')
x = input()
alg = Extrapolation(budget=5679576, qPairs = 2362497, delimiter='\t',  sourceFilePath=main_dir + 'sourceSample.tsv', targetFilePath=main_dir + 'targetSample.tsv', wScheme = 'JS_APPROX',  users_input=float(x))
alg.applyProcessing()
