from kde_based_algorithm import KDE_Based_Algorithm

main_dir = '../content/drive/MyDrive/s9/'
x=0
print('Enter desired recall:')
x = input()
sg = KDE_Based_Algorithm(budget = 5679576, qPairs = 2362497, delimiter='\t',  sourceFilePath=main_dir + 'sourceSample.tsv', targetFilePath=main_dir + 'targetSample.tsv', users_input=float(x))
sg.applyProcessing()
