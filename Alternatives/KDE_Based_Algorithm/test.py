#from extrapolation import Extrapolation

main_dir = '../content/drive/MyDrive/data/s1/'
x=0
print('Enter desired recall:')
x = input()
sg = KDE_Based_Algorithm(budget = 631064, qPairs = 312327, delimiter='\t',  sourceFilePath=main_dir + 'sourceSample.tsv', targetFilePath=main_dir + 'targetSample.tsv', users_input=float(x))
sg.applyProcessing()
