from giant import GIAnt

main_dir = '../content/drive/MyDrive/s1/'
sg = Extrapolation(qPairs = 312327, delimiter='\t',  sourceFilePath=main_dir + 'sourceSample.tsv', targetFilePath=main_dir + 'targetSample.tsv')
sg.applyProcessing()
