import os
import weka.core.jvm as jvm
from heuristics_algorithm import Heuristics_Algorithm


def setup_environment():
    os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-8-openjdk-amd64"
    os.environ["CLASSPATH"] = "weka/weka-3-8-5/weka.jar"
    jvm.start()

def main():
    main_dir = "/home/njdaras/Downloads/Geosfiles/data/"
    recall = input('Enter desired recall: ')
    
    sg = Heuristics_Algorithm(budget=5679576, qPairs=2362497, delimiter='\t',
                             sourceFilePath=os.path.join(main_dir, 'regions_gr.csv'),
                             targetFilePath=os.path.join(main_dir, 'wildlife_sanctuaries.csv'),
                             HeuristicCondition = "Dynamic_Qualifying_Distance_Threshold", ConditionLimit = 10000, 
                             DynamicFactor = 0.5, ViolationLimit = 0)
    sg.applyProcessing()

if __name__ == "__main__":
    setup_environment()
    main()


