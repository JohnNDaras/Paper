import math
import numpy as np
import random
import sys
import time
import pandas as pd
from collections import defaultdict
from shapely.geometry import LineString, MultiPolygon, Polygon
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, KFold, GroupKFold
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from utilities import CsvReader
from datamodel import RelatedGeometries

class KDE_Based_Algorithm:

    def __init__(self, budget: int, qPairs: int, delimiter: str, sourceFilePath: str, targetFilePath: str, users_input):
        self.users_input = users_input
        self.CLASS_SIZE = 500
        self.NO_OF_FEATURES = 16
        self.SAMPLE_SIZE = 100000
        self.POSITIVE_PAIR = 1
        self.NEGATIVE_PAIR = 0
        self.trainingPhase = False
        self.qPairs = qPairs

        self.budget = budget
        self.delimiter = delimiter
        self.sourceData = CsvReader.readAllEntities(delimiter, sourceFilePath)
        self.targetData = []
        print('Source geometries', len(self.sourceData))

        self.targetFilePath = targetFilePath
        self.datasetDelimiter = len(self.sourceData)
        self.relations = RelatedGeometries(qPairs)
        self.sample = []
        self.sample_for_verification = []
        self.spatialIndex = defaultdict(lambda: defaultdict(list))
        self.verifiedPairs = set()
        self.minimum_probability_threshold = 0
        self.thetaX = -1
        self.thetaY = -1

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

        self.targetData = CsvReader.readAllEntities(self.delimiter, self.targetFilePath)

        targetGeomId, self.allCandidatePairs = 0, 0
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

            no_of_points = self.getNoOfPoints(s)
            if self.maxFeatures[7] < no_of_points:
                self.maxFeatures[7] = no_of_points

            if no_of_points < self.minFeatures[7]:
                self.minFeatures[7] = no_of_points

            if self.maxFeatures[9] < targetGeom.length:
                self.maxFeatures[9] = targetGeom.length

            if targetGeom.length < self.minFeatures[9]:
                self.minFeatures[9] = targetGeom.length

            candidateMatches = self.getCandidates(targetGeomId, targetGeom)

            currentCandidates = 0
            currentDistinctCooccurrences = len(candidateMatches)
            currentCooccurrences = 0

            for candidateMatchId in candidateMatches:
              self.allCandidatePairs += 1
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

                  #Create sample for training
                  if len(self.sample) < self.SAMPLE_SIZE:
                        self.random_number = random.randint(0, 90)
                        if self.random_number == 0:
                          self.sample.append((candidateMatchId, targetGeomId, targetGeom))

                  #Create sample for verification
                  if len(self.sample_for_verification) < self.SAMPLE_SIZE:
                        if self.random_number == 1:
                          self.sample_for_verification.append((candidateMatchId, targetGeomId, targetGeom))

                  #self.allCandidatePairs += 1

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

        for i in range(self.datasetDelimiter):
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

    def getNoOfPoints(self, geometry):
        if isinstance(geometry, Polygon):
            return len(geometry.exterior.coords)
        elif isinstance(geometry, LineString):
            return len(geometry.coords)
        elif isinstance(geometry, MultiPolygon):
            return sum([len(polygon.exterior.coords) for polygon in geometry.geoms])
        else:
            #print(type(geometry))
            #print(geometry)
            return 0

    def getCandidates(self, targetId, targetGeom):
        candidates = set()

        envelope = targetGeom.envelope.bounds
        maxX = math.ceil(envelope[2] / self.thetaX)
        maxY = math.ceil(envelope[3] / self.thetaY)
        minX = math.floor(envelope[0] / self.thetaX)
        minY = math.floor(envelope[1] / self.thetaY)

        for latIndex in range(minX, maxX+1):
          for longIndex in range(minY,maxY+1):
              for sourceId in self.spatialIndex[latIndex][longIndex]:
                  if (self.flag[sourceId] == -1):
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


    @staticmethod
    def create_model(input_dim):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            BatchNormalization(),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        return model

    def trainModel(self):
      self.trainingPhase = True
      random.shuffle(self.sample)

      negativeClassFull, positiveClassFull = False, False
      negativePairs, positivePairs = [], []
      excessVerifications = 0

      for sourceId, targetId, targetGeom in self.sample:
          if negativeClassFull and positiveClassFull:
              break

          isRelated = self.relations.verifyRelations(sourceId, targetId, self.sourceData[sourceId], targetGeom)
          self.verifiedPairs.add((sourceId, targetId))

          if isRelated:
                if len(positivePairs) < self.CLASS_SIZE:
                    positivePairs.append((sourceId, targetId, targetGeom))
                else:
                    excessVerifications += 1
                    positiveClassFull = True
          else:
                if len(negativePairs) < self.CLASS_SIZE:
                    negativePairs.append((sourceId, targetId, targetGeom))
                else:
                    excessVerifications += 1
                    negativeClassFull = True

      # Prepare data for the neural network
      X, y = [], []
      for pair in negativePairs + positivePairs:
          sourceId, targetId, targetGeom = pair
          X.append(self.get_feature_vector(sourceId, targetId, targetGeom))
          y.append(1 if pair in positivePairs else 0)

      X = np.array(X)
      y = np.array(y)

      if len(negativePairs) == 0 or len(positivePairs) == 0:
          raise ValueError("Both negative and positive instances must be labelled.")

      # Create and compile the neural network model
      model = KDE_Based_Algorithm.create_model(X.shape[1])
      model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

      # Train the model
      model.fit(X, y, epochs=30, batch_size=32, validation_split=0.1, verbose=0, callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])

      self.classifier = model  # Store the trained model
      self.trainingPhase = False








    def get_feature_vector(self, sourceId, targetId, targetGeom):
        featureVector = [0] * (self.NO_OF_FEATURES)

        if(self.trainingPhase == 1):
          candidateMatches = self.getCandidates(targetId, targetGeom)
          for candidateMatchId in candidateMatches:
            featureVector[13] += self.frequency[candidateMatchId]
            featureVector[14]+=1
            if (self.validCandidate(candidateMatchId, targetGeom.envelope)): # intersecting MBRs
                  featureVector[15]+=1

        mbrIntersection = self.sourceData[sourceId].envelope.intersection(targetGeom.envelope)

        #area-based features
        featureVector[0] = (self.sourceData[sourceId].envelope.area - self.minFeatures[0]) / self.maxFeatures[0] * 10000  # source area
        featureVector[1] = (targetGeom.envelope.area - self.minFeatures[1]) / self.maxFeatures[1] * 10000  # target area
        featureVector[2] = (mbrIntersection.area - self.minFeatures[2]) / self.maxFeatures[2] * 10000  # intersection area

        #grid-based features
        featureVector[3] = (self.getNoOfBlocks(self.sourceData[sourceId].bounds) - self.minFeatures[3]) / self.maxFeatures[3] * 10000 # source blocks
        featureVector[4] = (self.getNoOfBlocks(targetGeom.bounds) - self.minFeatures[4]) / self.maxFeatures[4] * 10000 # source blocks
        featureVector[5] = (self.frequency[sourceId] - self.minFeatures[5]) / self.maxFeatures[5] * 10000 # common blocks

        # boundary-based features
        featureVector[7] = (self.getNoOfPoints(self.sourceData[sourceId]) - self.minFeatures[7]) / self.maxFeatures[7] * 10000 # source boundary points
        featureVector[8] = (self.getNoOfPoints(targetGeom) - self.minFeatures[8]) / self.maxFeatures[8] * 10000 # target boundary points
        featureVector[9] = (targetGeom.length - self.minFeatures[9]) / self.maxFeatures[9] * 10000 # source length
        featureVector[6] = (self.sourceData[sourceId].length - self.minFeatures[6]) / self.maxFeatures[6] * 10000  # target length
        #candidate-based features
        #source geometry
        featureVector[10] = (self.totalCooccurrences[sourceId] - self.minFeatures[10]) / self.maxFeatures[10] * 10000
        featureVector[11] = (self.distinctCooccurrences[sourceId] - self.minFeatures[11]) / self.maxFeatures[11] * 10000
        featureVector[12] = (self.realCandidates[sourceId] - self.minFeatures[12]) / self.maxFeatures[12] * 10000
        #target geometry
        featureVector[13] = (featureVector[13] - self.minFeatures[13]) / self.maxFeatures[13] * 10000
        featureVector[14] = (featureVector[14] - self.minFeatures[14]) / self.maxFeatures[14] * 10000
        featureVector[15] = (featureVector[15] - self.minFeatures[15]) / self.maxFeatures[15] * 10000

        return featureVector

    def getNoOfBlocks(self, envelope) :
      maxX = math.ceil(envelope[2] / self.thetaX)
      maxY = math.ceil(envelope[3] / self.thetaY)
      minX = math.floor(envelope[0] / self.thetaX)
      minY = math.floor(envelope[1] / self.thetaY)
      return (maxX - minX + 1) * (maxY - minY + 1)


    def classify_geometry_pair(self,geometry_pair):
        # Dictionary mapping pairs of geometry types to group numbers
        geometry_pairs = {
            (Point, Point): 1,
            (Point, LineString): 2,
            (Point, Polygon): 3,
            (LineString, LineString): 4,
            (LineString, Polygon): 5,
            (Polygon, Polygon): 6,
            (MultiPoint, MultiPoint): 7,
            (MultiLineString, MultiLineString): 8,
            (MultiPolygon, MultiPolygon): 9,
            (MultiPoint, LineString): 10,
            (MultiLineString, Polygon): 11,
        }

        # Unpack the tuple into the two geometry types
        geom1, geom2 = type(geometry_pair[0]), type(geometry_pair[1])

        # Check if the pair (in any order) exists in the dictionary
        if (geom1, geom2) in geometry_pairs:
            return geometry_pairs[(geom1, geom2)]
        elif (geom2, geom1) in geometry_pairs:
            return geometry_pairs[(geom2, geom1)]
        else:
            return None  # Return None or raise an error if the pair is not found



    def verification(self):
        Prediction_probs, retainedPairs = [], []
        instances = []
        validcandidates = []
        self.groups_by_geometry_shape = []
        targetId, totalDecisions, positiveDecisions, truePositiveDecisions = 0, 0, 0, 0
        counter = 0
        group_number = 0
        self.relations.reset()
        for candidateMatchId, targetGeomId, targetGeom in self.sample_for_verification:

          if positiveDecisions == 15000:
            break
          currentInstance = self.get_feature_vector(candidateMatchId, targetGeomId, targetGeom)
          isRelated = self.relations.verifyRelations(candidateMatchId, targetId, self.sourceData[candidateMatchId], targetGeom)
          if isRelated:
              positiveDecisions += 1
              instances.append((currentInstance, candidateMatchId, targetGeomId, targetGeom))



        # Batch predict
        if instances:
            features, indices = zip(*[(instance[0], instance[1:]) for instance in instances])
            features = np.array(features)
            predictions = self.classifier.predict(features)



            for pred, idx in zip(predictions, indices):
                weight = float(pred[0])
                Prediction_probs.append(weight)
                group_number = self.classify_geometry_pair((self.sourceData[idx[0]], idx[2]))
                #print(type(self.sourceData[idx[0]]), type(idx[2]),group_number)
                self.groups_by_geometry_shape.append(group_number)


        self.relations.reset()
        Prediction_probs = pd.DataFrame({'0': Prediction_probs})
        Prediction_probs = Prediction_probs['0']
        kde_model2 = self.get_best_model(Prediction_probs)
        self.find_estimate_threshold(kde_model2)



        for targetGeom in self.targetData:
            candidates = self.getCandidates(targetId, targetGeom)
            for candidateMatchId in candidates:
                if self.validCandidate(candidateMatchId, targetGeom.envelope):
                    currentInstance = self.get_feature_vector(candidateMatchId, targetId, targetGeom)
                    validcandidates.append((currentInstance, candidateMatchId, targetId, targetGeom))
            targetId += 1

        self.targetData.clear()
        # Batch predict
        if validcandidates:
            features, indices = zip(*[(instance[0], instance[1:]) for instance in validcandidates])
            features = np.array(features)
            predictions = self.classifier.predict(features)
            for pred, idx in zip(predictions, indices):
                weight = float(pred[0])
                if weight >= self.minimum_probability_threshold:
                    retainedPairs.append((weight, idx))

        for weight, (candidateMatchId, targetId, targetGeom) in retainedPairs:
            counter += 1
            if self.relations.verifyRelations(candidateMatchId, targetId, self.sourceData[candidateMatchId], targetGeom):
                truePositiveDecisions += 1
            if (self.budget == counter):
                break

        print("True Positive Decisions\t:\t" + str(truePositiveDecisions))


    def create_groups(self, probabilities):
        # This assumes `probabilities` is a numpy array of float values between 0 and 1.
        groups = np.floor(probabilities * 10) % 10  # Extract the first digit after the decimal
        return groups.astype(int)  # Convert to integer for use as group labels


    def get_best_model(self, x_train, samples=200, h_vals=np.arange(0.001, 0.21, 0.01), seed=42):
        # Using a smaller number of bandwidths and KFold cross-validation
        groups_by_probability = self.create_groups(x_train)                                # group by probability
        self.groups_by_geometry_shape = np.array(self.groups_by_geometry_shape)            # group by geometry type
        groups = self.groups_by_geometry_shape + groups_by_probability / 10.0              # group by probability and geometry type
        splits = set(groups)
        # Count the number of distinct items
        number_of_splits = len(splits)
        kernels = ['cosine', 'epanechnikov', 'gaussian', 'linear', 'tophat', 'exponential']
        print("Testing {} options with Grid Search".format(len(h_vals)*len(kernels)))
        grid = GridSearchCV(
            KernelDensity(),
            {'bandwidth': h_vals, 'kernel': kernels},
            cv=GroupKFold(n_splits = number_of_splits),  # Using GroupKFold
            n_jobs=-1  # Utilize all available CPU cores for the grid search
        )
        grid.fit(np.expand_dims(x_train, axis=1), groups=groups)  # Fit model on the reshaped data
        self.visualize_diagnostics(x_train)  # Visualize the distribution of the training data and the KDE fit
        return grid.best_estimator_


    def find_estimate_threshold(self,model, interpolation_points=1000):
      self.simaia = False
      estimations = []
      values = np.linspace(0.0001, 1.0, interpolation_points)
      for threshold in np.arange(0,1.0,0.02):
          est = self.compute_estimate_cdf(model, target_range=(0, threshold))
          print(threshold, 1 - est)
          estimations.append((threshold,1-est))
      self.minimum_probability_threshold = self.find_closest(self.users_input, estimations)
      print("This is minimum ",self.minimum_probability_threshold)
      return threshold, est

    def compute_estimate_cdf(self,model, target_range=(0, 1), interpolation_points=5000, margin=0.01):
        x_test, log_dens = self.get_logs(model, target_range, interpolation_points, margin)
        probs = np.exp(log_dens)
        # Estimate AUC
        auc = np.trapz(probs.ravel(), x_test.ravel())
        return auc


    def generate_test_interval(self,target_range=(0, 1), interpolation_points=1000, margin=0.01):
        start = target_range[0]-target_range[0]*margin
        stop = target_range[1]+target_range[1]*margin
        x_test = np.linspace(start, stop, interpolation_points)[:, np.newaxis]
        return x_test


    def get_logs(self,model, target_range=(0, 1), interpolation_points=1000, margin=0.01):
        x_test = self.generate_test_interval(target_range, interpolation_points, margin)
        #log_dens = self.scorer(model,x_test)
        log_dens = model.score_samples(x_test)
        return x_test, log_dens

    def find_closest(self,target, tuples_list):
        closest_tuple = None
        min_difference = float('inf')

      #  target = target - 0.1  
        for first_num, second_num in tuples_list:
            difference = abs(target - second_num)
            if difference < min_difference:
                min_difference = difference
                threshold = first_num
        return threshold

    def visualize_diagnostics(self, data):
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=30, density=True, alpha=0.5, label='Histogram of Data')
        kde = gaussian_kde(data)
        x_range = np.linspace(min(data), max(data), 1000)
        plt.plot(x_range, kde(x_range), 'r-', label='KDE')
        plt.title('Histogram and KDE Overlay')
        plt.xlabel('Data Values')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()
