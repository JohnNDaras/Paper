import math
import numpy as np
import random
import sys
import time
import shapely
import os
import random
import weka.core.jvm as jvm
from collections import defaultdict
from weka.core.dataset import Instances, Attribute, Instance
from weka.classifiers import Classifier
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from utilities import CsvReader
from datamodel import RelatedGeometries

class KDE_Based_Algorithm:

    def __init__(self, budget: int, qPairs: int, delimiter: str, sourceFilePath: str, targetFilePath: str, target_recall):
        self.CLASS_SIZE = 500
        self.NO_OF_FEATURES = 16
        self.SAMPLE_SIZE = 50000
        self.POSITIVE_PAIR = 1
        self.NEGATIVE_PAIR = 0
        self.budget = budget
        self.target_recall = target_recall

        self.sourceData = CsvReader.readAllEntities(delimiter, sourceFilePath)
        print('Source geometries', len(self.sourceData))
        self.targetData = CsvReader.readAllEntities(delimiter, targetFilePath)
        print('Target geometries', len(self.targetData))

        self.kde_sample = set()
        self.predicted_probabilities = []
        self.relations = RelatedGeometries(qPairs)
        self.sample = set()
        self.spatialIndex = defaultdict(lambda: defaultdict(list))
        self.verifiedPairs = set()

    def applyProcessing(self) :
      time1 = int(time.time() * 1000)
      self.setThetas()
      self.indexSource()
      time2 = int(time.time() * 1000)
      self.preprocessing()
      time3 = int(time.time() * 1000)
      self.trainModel()
      time4 = int(time.time() * 1000)
      self.verification()
      time5 = int(time.time() * 1000)

      print("Indexing Time\t:\t" + str(time2 - time1))
      print("Initialization Time\t:\t" + str(time3 - time2))
      print("Training Time\t:\t" + str(time4 - time3))
      print("Verification Time\t:\t" + str(time5 - time4))
      self.relations.print()

    def indexSource(self) :
      geometryId = 0
      for sEntity in self.sourceData:
        self.addToIndex(geometryId, sEntity.bounds)
        geometryId += 1

    def addToIndex(self, geometryId, envelope) :
        maxX = math.ceil(envelope[2] / self.thetaX)
        maxY = math.ceil(envelope[3] / self.thetaY)
        minX = math.floor(envelope[0] / self.thetaX)
        minY = math.floor(envelope[1] / self.thetaY)
        for latIndex in range(minX, maxX+1):
          for longIndex in range(minY, maxY+1):
              self.spatialIndex[latIndex][longIndex].append(geometryId)

    def preprocessing(self):
        sampled_ids = set()
        max_candidates = 10 * len(self.sourceData)
        while (len(sampled_ids) < self.SAMPLE_SIZE):
            sampled_ids.add(random.randint(0, max_candidates))

        kde_sample_ids = set()
        while (len(kde_sample_ids) < 2000):
            kde_sample_ids.add(random.randint(0, max_candidates))

        self.flag = [-1] * len(self.sourceData)
        self.frequency = [-1] * len(self.sourceData)
        self.distinctCooccurrences =  [0] * len(self.sourceData)
        self.realCandidates =  [0] * len(self.sourceData)
        self.totalCooccurrences =  [0] * len(self.sourceData)
        self.maxFeatures = [-sys.float_info.max] * self.NO_OF_FEATURES
        self.minFeatures = [sys.float_info.max] * self.NO_OF_FEATURES

        for s in self.sourceData:
            if self.maxFeatures[0] < s.envelope.area:
                self.maxFeatures[0] = s.envelope.area

            if s.envelope.area < self.minFeatures[0]:
                self.minFeatures[0] = s.envelope.area

            no_of_blocks = self.getNoOfBlocks(s.bounds)
            if self.maxFeatures[3] < no_of_blocks:
                self.maxFeatures[3] = no_of_blocks

            if no_of_blocks < self.minFeatures[3]:
                self.minFeatures[3] = no_of_blocks

            no_of_points = self.getNoOfPoints(s)

            if self.maxFeatures[6] < no_of_points:
                self.maxFeatures[6] = no_of_points

            if no_of_points < self.minFeatures[6]:
                self.minFeatures[6] = no_of_points

            if self.maxFeatures[8] < s.length:
                self.maxFeatures[8] = s.length

            if s.length < self.minFeatures[8]:
                self.minFeatures[8] = s.length

        targetGeomId, pairId = 0, 0
        for targetGeom in self.targetData:
            if self.maxFeatures[1] < targetGeom.envelope.area:
                self.maxFeatures[1] = targetGeom.envelope.area

            if targetGeom.envelope.area < self.minFeatures[1]:
                self.minFeatures[1] = targetGeom.envelope.area

            noOfBlocks = self.getNoOfBlocks(targetGeom.bounds)
            if self.maxFeatures[4] < noOfBlocks:
                self.maxFeatures[4] = noOfBlocks

            if noOfBlocks < self.minFeatures[4]:
                self.minFeatures[4] = noOfBlocks

            no_of_points = self.getNoOfPoints(targetGeom)

            if self.maxFeatures[7] < no_of_points:
                self.maxFeatures[7] = no_of_points

            if no_of_points < self.minFeatures[7]:
                self.minFeatures[7] = no_of_points

            if self.maxFeatures[9] < targetGeom.length:
                self.maxFeatures[9] = targetGeom.length

            if targetGeom.length < self.minFeatures[9]:
                self.minFeatures[9] = targetGeom.length

            candidateMatches = self.getCandidates(targetGeomId)

            currentCandidates = 0
            currentDistinctCooccurrences = len(candidateMatches)
            currentCooccurrences = 0
            for candidateMatchId in candidateMatches:
              self.distinctCooccurrences[candidateMatchId] += 1
              currentCooccurrences += self.frequency[candidateMatchId]

              self.totalCooccurrences[candidateMatchId] += self.frequency[candidateMatchId]

              if self.validCandidate(candidateMatchId, targetGeom.envelope):
                  currentCandidates += 1
                  self.realCandidates[candidateMatchId] += 1

                  mbrIntersection = self.sourceData[candidateMatchId].envelope.intersection(targetGeom.envelope)

                  if self.maxFeatures[2] < mbrIntersection.area:
                      self.maxFeatures[2] = mbrIntersection.area

                  if mbrIntersection.area < self.minFeatures[2]:
                      self.minFeatures[2] = mbrIntersection.area

                  if self.maxFeatures[5] < self.frequency[candidateMatchId]:
                      self.maxFeatures[5] =  self.frequency[candidateMatchId]

                  if self.frequency[candidateMatchId] < self.minFeatures[5]:
                      self.minFeatures[5] = self.frequency[candidateMatchId]

                  if (pairId in sampled_ids):
                      self.sample.add((candidateMatchId, targetGeomId))

                  if (pairId in kde_sample_ids):
                      self.kde_sample.add((candidateMatchId, targetGeomId))

                  pairId += 1

            if self.maxFeatures[13] < currentCooccurrences:
                self.maxFeatures[13] = currentCooccurrences

            if currentCooccurrences < self.minFeatures[13]:
                self.minFeatures[13] = currentCooccurrences

            if self.maxFeatures[14] < currentDistinctCooccurrences:
                self.maxFeatures[14] = currentDistinctCooccurrences

            if currentDistinctCooccurrences < self.minFeatures[14]:
                self.minFeatures[14] = currentDistinctCooccurrences

            if self.maxFeatures[15] < currentCandidates:
                self.maxFeatures[15] = currentCandidates

            if currentCandidates < self.minFeatures[15]:
                self.minFeatures[15] = currentCandidates

            targetGeomId += 1

        for i in range(len(self.sourceData)):
            if self.maxFeatures[10] < self.totalCooccurrences[i]:
                self.maxFeatures[10] = self.totalCooccurrences[i]

            if self.totalCooccurrences[i] < self.minFeatures[10]:
                self.minFeatures[10] = self.totalCooccurrences[i]

            if self.maxFeatures[11] < self.distinctCooccurrences[i]:
                self.maxFeatures[11] = self.distinctCooccurrences[i]

            if self.distinctCooccurrences[i] < self.minFeatures[11]:
                self.minFeatures[11] = self.distinctCooccurrences[i]

            if self.maxFeatures[12] < self.realCandidates[i]:
                self.maxFeatures[12] = self.realCandidates[i]

            if self.realCandidates[i] < self.minFeatures[12]:
                self.minFeatures[12] = self.realCandidates[i]


    def getNoOfPoints(self, geometries):
        # Using vectorized get_num_coordinates to count points in each geometry
        return get_num_coordinates(geometries)


    def getCandidates(self, targetId):
        candidates = set()

        targetGeom = self.targetData[targetId]
        envelope = targetGeom.envelope.bounds
        maxX = math.ceil(envelope[2] / self.thetaX)
        maxY = math.ceil(envelope[3] / self.thetaY)
        minX = math.floor(envelope[0] / self.thetaX)
        minY = math.floor(envelope[1] / self.thetaY)

        for latIndex in range(minX, maxX+1):
          for longIndex in range(minY,maxY+1):
              for sourceId in self.spatialIndex[latIndex][longIndex]:
                  if (self.flag[sourceId] != targetId): #!!!!!!THIS LINE WAS DEBUGGED
                      self.flag[sourceId] = targetId
                      self.frequency[sourceId] = 0
                  self.frequency[sourceId] += 1
                  candidates.add(sourceId)

        return candidates

    def setThetas(self):
        self.thetaX, self.thetaY = 0, 0
        for sEntity in self.sourceData:
            envelope = sEntity.envelope.bounds
            self.thetaX += envelope[2] - envelope[0]
            self.thetaY += envelope[3] - envelope[1]

        self.thetaX /= len(self.sourceData)
        self.thetaY /= len(self.sourceData)
        print("Dimensions of Equigrid", self.thetaX,"and", self.thetaY)

    def validCandidate(self, candidateId, targetEnv):
        return self.sourceData[candidateId].envelope.intersects(targetEnv)


    def trainModel(self):
        self.sample = list(self.sample)
        random.shuffle(self.sample)
        positiveSourceIds, positiveTargetIds = [], []
        negativeSourceIds, negativeTargetIds = [], []

        positiveFeatures, negativeFeatures = [], []
        excesive_positive_verifications = set()
        negativePairs, positivePairs, excessVerifications = 0, 0, 0
        for sourceId, targetId in self.sample:
          if negativePairs == self.CLASS_SIZE and positivePairs == self.CLASS_SIZE:
              break

          isRelated = self.relations.verifyRelations(sourceId, targetId, self.sourceData[sourceId], self.targetData[targetId])
          self.verifiedPairs.add((sourceId, targetId))

          if isRelated:
                if positivePairs < self.CLASS_SIZE:
                    positivePairs += 1
                    positiveSourceIds.append(sourceId)
                    positiveTargetIds.append(targetId)
                else:
                    excessVerifications += 1
                    excesive_positive_verifications.add((sourceId, targetId))
          else:
                if negativePairs < self.CLASS_SIZE:
                    negativePairs += 1
                    negativeSourceIds.append(sourceId)
                    negativeTargetIds.append(targetId)
                else:
                    excessVerifications += 1

        positiveFeatures = self.get_feature_vector(positiveSourceIds, positiveTargetIds).tolist()
        negativeFeatures = self.get_feature_vector(negativeSourceIds, negativeTargetIds).tolist()
        print("Positive Verifications\t:\t" + str(positivePairs))
        print("Negative Verifications\t:\t" + str(negativePairs))
        print("Excess Verifications\t:\t" + str(excessVerifications))
        #print(type(positiveFeatures))
        #print(positiveFeatures)

        # Create an Instances object with the specified attributes
        number_of_features = len(positiveFeatures[0])
        print(number_of_features)
        print(len(positiveFeatures))
        attrs = [Attribute.create_numeric(f"feature{i}") for i in range(number_of_features)]
        attrs.append(Attribute.create_nominal("class", ["positive", "negative"]))

        # Create a dataset with a name 'example' and the defined attributes
        self.dataset = Instances.create_instances(name="example", atts=attrs, capacity=len(positiveFeatures) + len(negativeFeatures))
        self.dataset.class_is_last()  # Setting the class attribute

        # Adding positive feature data to the dataset
        for features in positiveFeatures:
            values = features + [1]  # 'positive' is 1
            inst = Instance.create_instance(values)
            self.dataset.add_instance(inst)

        # Adding negative feature data to the dataset
        for features in negativeFeatures:
            values = features + [0]  # 'negative' is 0
            inst = Instance.create_instance(values)
            self.dataset.add_instance(inst)

        # Initialize the classifier
        self.classifier = Classifier(classname="weka.classifiers.functions.Logistic")

        #print(self.dataset)
        # Build the classifier with the dataset
        self.classifier.build_classifier(self.dataset)


    def get_feature_vector(self, sourceIds, targetIds):
        featureVectors = np.zeros((len(sourceIds), self.NO_OF_FEATURES))

        sourceGeometries = np.array([self.sourceData[sourceId] for sourceId in sourceIds])
        targetGeometries = np.array([self.targetData[targetId] for targetId in targetIds])


        #Get Envelopes
        SourceGeomEnvelopes = shapely.envelope(sourceGeometries)
        TargetGeomEnvelopes = shapely.envelope(targetGeometries)

        #Get Envelope Bounds
        SourceGeomEnvelopesBounds = shapely.bounds(SourceGeomEnvelopes)
        TargetGeomEnvelopesBounds = shapely.bounds(TargetGeomEnvelopes)

        #MBR Intersection Area
        pairs = list(zip(SourceGeomEnvelopes, TargetGeomEnvelopes))
        mbrIntersection = shapely.area(shapely.intersection(*np.transpose(pairs)))

        #Geometries Envelopes Areas
        SourceGeomAreas = shapely.area(SourceGeomEnvelopes)
        TargetGeomAreas = shapely.area(TargetGeomEnvelopes)

        #Geometries Length
        SourceGeomLenght = shapely.length(sourceGeometries)
        TargetGeomLenght = shapely.length(targetGeometries)

        #Get Number of Blocks of Geometries
        sourceBounds = shapely.bounds(sourceGeometries)
        targetBounds = shapely.bounds(targetGeometries)
        SourceBlocks = self.getNoOfBlocks1(sourceBounds)
        TargetBlocks = self.getNoOfBlocks1(targetBounds)


        #Get Number Of Points
        source_no_of_points = self.getNoOfPoints(sourceGeometries)
        target_no_of_points = self.getNoOfPoints(targetGeometries)

        for i, (sourceGeom, targetGeom) in enumerate(zip(sourceIds, targetIds)):

            candidateMatches = self.getCandidates(targetGeom)

            featureVectors[i, 13] = sum(self.frequency[candidateMatchId] for candidateMatchId in candidateMatches)
            featureVectors[i, 14] = len(candidateMatches)

            #validCandidateCount = sum(1 for candidateMatchId in candidateMatches if self.validCandidate(candidateMatchId, TargetGeomEnvelopes[i]))

            if mbrIntersection[i] > 0:
              featureVectors[i, 15] += 1

            # Area-based features
            featureVectors[i, 0] = (SourceGeomAreas[i] - self.minFeatures[0]) / self.maxFeatures[0] * 10000  # source area
            featureVectors[i, 1] = (TargetGeomAreas[i] - self.minFeatures[1]) / self.maxFeatures[1] * 10000  # target area
            featureVectors[i, 2] = (mbrIntersection[i] - self.minFeatures[2]) / self.maxFeatures[2] * 10000  # intersection area

            # Grid-based features
            featureVectors[i, 3] = (SourceBlocks[i] - self.minFeatures[3]) / self.maxFeatures[3] * 10000  # source blocks
            featureVectors[i, 4] = (TargetBlocks[i] - self.minFeatures[4]) / self.maxFeatures[4] * 10000  # target blocks
            featureVectors[i, 5] = (self.frequency[sourceIds[i]] - self.minFeatures[5]) / self.maxFeatures[5] * 10000  # common blocks

            # Boundary-based features
            featureVectors[i, 6] = (source_no_of_points[i] - self.minFeatures[6]) / self.maxFeatures[6] * 10000  # source boundary points
            featureVectors[i, 7] = (target_no_of_points[i] - self.minFeatures[7]) / self.maxFeatures[7] * 10000  # target boundary points
            featureVectors[i, 8] = (SourceGeomLenght[i] - self.minFeatures[8]) / self.maxFeatures[8] * 10000  # source length
            featureVectors[i, 9] = (TargetGeomLenght[i] - self.minFeatures[9]) / self.maxFeatures[9] * 10000  # target length

            # Candidate-based features
            # Source geometry
            featureVectors[i, 10] = (self.totalCooccurrences[sourceIds[i]] - self.minFeatures[10]) / self.maxFeatures[10] * 10000
            featureVectors[i, 11] = (self.distinctCooccurrences[sourceIds[i]] - self.minFeatures[11]) / self.maxFeatures[11] * 10000
            featureVectors[i, 12] = (self.realCandidates[sourceIds[i]] - self.minFeatures[12]) / self.maxFeatures[12] * 10000
            # Target geometry
            featureVectors[i, 13] = (featureVectors[i, 13] - self.minFeatures[13]) / self.maxFeatures[13] * 10000
            featureVectors[i, 14] = (featureVectors[i, 14] - self.minFeatures[14]) / self.maxFeatures[14] * 10000
            featureVectors[i, 15] = (featureVectors[i, 15] - self.minFeatures[15]) / self.maxFeatures[15] * 10000

        return featureVectors



    def getNoOfBlocks(self, envelope) :
      maxX = math.ceil(envelope[2] / self.thetaX)
      maxY = math.ceil(envelope[3] / self.thetaY)
      minX = math.floor(envelope[0] / self.thetaX)
      minY = math.floor(envelope[1] / self.thetaY)
      return (maxX - minX + 1) * (maxY - minY + 1)

    def getNoOfBlocks1(self, envelopes):
        blocks = []
        for envelope in envelopes:
            maxX = math.ceil(envelope[2] / self.thetaX)
            maxY = math.ceil(envelope[3] / self.thetaY)
            minX = math.floor(envelope[0] / self.thetaX)
            minY = math.floor(envelope[1] / self.thetaY)
            blocks.append((maxX - minX + 1) * (maxY - minY + 1))
        return blocks

    def verification(self):
        totalDecisions, truePositiveDecisions = len(self.verifiedPairs), 0

        self.kde_sample = list(self.kde_sample)
        random.shuffle(self.kde_sample)
        SourceInstanceIndexes, TargetInstanceIndexes = [], []

        #self.relations.reset() !!! CHECK WHETHER IT IS NEEDED
        KDEtargetGeometries = np.array([self.targetData[targetGeomId] for sourceGeomId, targetGeomId in self.kde_sample])
        KDETargetGeomEnvelopes = shapely.envelope(KDEtargetGeometries)
        for i, (sourceGeomId, targetGeomId) in enumerate(self.kde_sample):
            if (self.validCandidate(sourceGeomId, KDETargetGeomEnvelopes[i])):
              if (sourceGeomId, targetGeomId) in self.verifiedPairs:
                continue

              totalDecisions += 1
              isRelated = self.relations.verifyRelations(sourceGeomId, targetGeomId, self.sourceData[sourceGeomId], self.targetData[targetGeomId])
              if isRelated:
                  SourceInstanceIndexes.append(sourceGeomId)
                  TargetInstanceIndexes.append(targetGeomId)

        Instances = self.get_feature_vector(SourceInstanceIndexes, TargetInstanceIndexes).tolist()
        for currentInstance in Instances:
            new_instance = Instance.create_instance(currentInstance + [None])  # No class value
            new_instance.dataset = self.dataset
            probabilities = self.classifier.distribution_for_instance(new_instance)
            self.predicted_probabilities.append(probabilities[1])

        #self.predicted_probabilities.sort()
        print(self.predicted_probabilities)
        print('KDE verifications', totalDecisions - len(self.verifiedPairs))
        print('KDE positive verifications', len(self.predicted_probabilities))

        kde_model2 = self.get_best_model(self.predicted_probabilities)
        self.find_estimate_threshold(kde_model2)

        SourceInstanceIndexes.clear()
        TargetInstanceIndexes.clear()
        Instances.clear()

        counter = 0
        for targetId in range(len(self.targetData)):
          #candidateMatches = self.getCandidates(targetId)
          candidateMatches = self.getCandidates(targetId)
          for candidateMatchId in candidateMatches:
              if (self.validCandidate(candidateMatchId, self.targetData[targetId].envelope)):
                if (candidateMatchId, targetId) in self.verifiedPairs:
                  continue

                totalDecisions += 1

                SourceInstanceIndexes.append(candidateMatchId)
                TargetInstanceIndexes.append(targetId)

        print("SourceInstanceIndexes", len(SourceInstanceIndexes))
        print("TargetInstanceIndexes", len(TargetInstanceIndexes))
        Instances = self.get_feature_vector(SourceInstanceIndexes, TargetInstanceIndexes).tolist()
        for currentInstance in Instances:
                new_instance = Instance.create_instance(currentInstance + [None])  # No class value
                new_instance.dataset = self.dataset
                probabilities = self.classifier.distribution_for_instance(new_instance)
                if probabilities[1] >= self.minimum_probability_threshold:
                #    counter = counter + 1
                    if (self.budget == counter):
                      break
                    if self.relations.verifyRelations(SourceInstanceIndexes[counter], TargetInstanceIndexes[counter], self.sourceData[SourceInstanceIndexes[counter]], self.targetData[TargetInstanceIndexes[counter]]):
                      truePositiveDecisions += 1
                counter = counter + 1
        print("True Positive Decisions\t:\t" + str(truePositiveDecisions))

    def get_best_model(self, x_train, samples=200, h_vals=np.arange(0.001, 0.21, 0.01), seed=42):
        kernels = ['cosine', 'epanechnikov', 'gaussian', 'linear', 'tophat', 'exponential']
        print("Testing {} options with Grid Search".format(len(h_vals)*len(kernels)))
        grid = GridSearchCV(KernelDensity(), {'bandwidth': h_vals, 'kernel': kernels}, cv=LeaveOneOut())
        grid.fit(np.expand_dims(x_train, axis=1))
        print('Best KDE estimator', grid.best_estimator_)
        return grid.best_estimator_

    def find_estimate_threshold(self,model, interpolation_points=1000):
      estimations = []
      for threshold in np.arange(0,1.0,0.02):
          est = self.compute_estimate_cdf(model, target_range=(0, threshold))
          print(threshold, 1 - est)
          estimations.append((threshold,1-est))
      self.minimum_probability_threshold = self.find_closest(self.target_recall, estimations)
      print("This is minimum ", self.minimum_probability_threshold)

    def compute_estimate_cdf(self,model, target_range=(0, 1), interpolation_points=1000, margin=0.01):
        x_test, log_dens = self.get_logs(model, target_range, interpolation_points, margin)
        probs = np.exp(log_dens)
        auc = np.trapz(probs.ravel(), x_test.ravel())
        return auc

    def generate_test_interval(self,target_range=(0, 1), interpolation_points=1000, margin=0.01):
        start = target_range[0]-target_range[0]*margin
        stop = target_range[1]+target_range[1]*margin
        x_test = np.linspace(start, stop, interpolation_points)[:, np.newaxis]
        return x_test

    def get_logs(self,model, target_range=(0, 1), interpolation_points=1000, margin=0.01):
        x_test = self.generate_test_interval(target_range, interpolation_points, margin)
        log_dens = model.score_samples(x_test)
        return x_test, log_dens

    def find_closest(self,target, tuples_list):
        min_difference = float('inf')
        for first_num, second_num in tuples_list:
            difference = abs(target - second_num)
            if difference < min_difference:
                min_difference = difference
                threshold = first_num
        return threshold

