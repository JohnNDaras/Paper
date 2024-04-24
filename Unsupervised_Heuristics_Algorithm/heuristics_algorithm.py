import math
import random
import time
from collections import defaultdict
from sortedcontainers import SortedList
from utilities import CsvReader
from datamodel import RelatedGeometries

# Constants for the application
MINIMUM_WEIGHT = 0.1
RANDOM_THRESHOLD = 15
SAMPLE_SIZE = 1000

class Heuristics_Algorithm:

    def __init__(self, budget,  qPairs,  delimiter,  sourceFilePath,  targetFilePath, wScheme, HeuristicCondition: str, ConditionLimit, DynamicFactor, ViolationLimit):
        self.heuristicCondition = HeuristicCondition
        self.condition_limit = ConditionLimit
        self.dynamic_factor = DynamicFactor
        self.violation_limit = ViolationLimit
        self.detectedQP = 0
        self.totalCandidatePairs = 0
        self.sample = []
        self.budget = budget
        self.delimiter = delimiter
        self.relations = RelatedGeometries(qPairs)
        self.sourceData = CsvReader.readAllEntities(delimiter, sourceFilePath)
        self.spatialIndex = defaultdict(lambda: defaultdict(list))
        self.targetFilePath = targetFilePath
        self.thetaX = -1
        self.thetaY = -1
        self.wScheme = wScheme
        self.sorted_list = SortedList()

    def getMethodName(self):
        return 'Heuristics Algorithm with Unsupervised Scheduling'

    def addToIndex(self, geometryId, envelope):
        grid_indices = self.compute_grid_indices(envelope)
        for latIndex in range(grid_indices['minX'], grid_indices['maxX']):
            for longIndex in range(grid_indices['minY'], grid_indices['maxY']):
                self.spatialIndex[latIndex][longIndex].append(geometryId)

    def compute_grid_indices(self, envelope):
        maxX = math.ceil(envelope[2] / self.thetaX)
        maxY = math.ceil(envelope[3] / self.thetaY)
        minX = math.floor(envelope[0] / self.thetaX)
        minY = math.floor(envelope[1] / self.thetaY)
        return {'maxX': maxX, 'maxY': maxY, 'minX': minX, 'minY': minY}

    def applyProcessing(self):
        time1 = int(time.time() * 1000)
        self.filtering()
        time2 = int(time.time() * 1000)
        self.initialization()
        time3 = int(time.time() * 1000)
        self.verification()
        time4 = int(time.time() * 1000)
        self.indexingTime = time2 - time1
        self.initializationTime = time3 - time2
        self.verificationTime = time4 - time3
        self.printResults()

    def filtering(self):
        self.setThetas()
        self.indexSource()

    def getCandidates(self, targetId, tEntity):
        candidates = set()
        envelope = tEntity.envelope.bounds
        grid_indices = self.compute_grid_indices(envelope)
        for latIndex in range(grid_indices['minX'], grid_indices['maxX']+1):
            for longIndex in range(grid_indices['minY'], grid_indices['maxY']+1):
                for sourceId in self.spatialIndex[latIndex][longIndex]:
                    if self.flag[sourceId] == -1:
                        self.flag[sourceId] = targetId
                    self.freq[sourceId] += 1
                    candidates.add(sourceId)
        return candidates


    def getNoOfBlocks(self,envelope) :
      maxX = math.ceil(envelope[2] / self.thetaX)
      maxY = math.ceil(envelope[3] / self.thetaY)
      minX = math.floor(envelope[0] / self.thetaX)
      minY = math.floor(envelope[1] / self.thetaY)
      return (maxX - minX + 1) * (maxY - minY + 1)


    def getWeight(self, sourceId, tEntity):
        """Calculate the weight between source and target entities based on the selected weighting scheme."""
        if self.wScheme == 'CF':
            return self.commonFraction(sourceId, tEntity)
        elif self.wScheme == 'JS_APPROX':
            return self.jaccardApproximation(sourceId, tEntity)
        elif self.wScheme == 'MBR':
            return self.minimumBoundingRectangle(sourceId, tEntity)
        elif self.wScheme == 'DISTANCE':
            return self.distanceBasedWeight(sourceId, tEntity)
        else:
            raise ValueError("Unknown weighting scheme")

    def commonFraction(self, sourceId, tEntity):
        commonBlocks = self.freq[sourceId]
        return commonBlocks

    def jaccardApproximation(self, sourceId, tEntity):
        totalBlocks = self.getNoOfBlocks(self.sourceData[sourceId].bounds) + self.getNoOfBlocks(tEntity.bounds) - self.freq[sourceId]
        return self.freq[sourceId] / totalBlocks if totalBlocks > 0 else 0

    def minimumBoundingRectangle(self, sourceId, tEntity):
        srcEnv = self.sourceData[sourceId].envelope
        trgEnv = tEntity.envelope
        mbrIntersection = srcEnv.intersection(trgEnv)
        denominator = srcEnv.area + trgEnv.area - mbrIntersection.area
        return mbrIntersection.area / denominator if denominator > 0 else 0

    def distanceBasedWeight(self, sourceId, tEntity):
        srcCenter = self.sourceData[sourceId].envelope.centroid
        trgCenter = tEntity.envelope.centroid
        distance = srcCenter.distance(trgCenter)
        # Normalize or scale the distance as needed
        return 1 / (1 + distance)  # Example: inverse distance weighting



    def indexSource(self) :
      geometryId = 0
      for sEntity in self.sourceData:
        self.addToIndex(geometryId, sEntity.bounds)
        geometryId += 1


    def initialization(self):
        self.flag = [-1] * len(self.sourceData)
        self.freq = [0] * len(self.sourceData)
        targetData = CsvReader.readAllEntities(self.delimiter, self.targetFilePath)
        targetId = 0

        for targetGeom in targetData:
            candidates = self.getCandidates(targetId, targetGeom)
            for candidateMatchId in candidates:
                if self.validCandidate(candidateMatchId, targetGeom.envelope):
                    self.totalCandidatePairs += 1
                    weight = self.getWeight(candidateMatchId, targetGeom)
                    if weight >= MINIMUM_WEIGHT:
                        self.sorted_list.add((weight, (candidateMatchId, targetId, targetGeom)))
                        if len(self.sorted_list) > self.budget:
                            self.sorted_list.pop(0)
            targetId += 1
        print("Total target geometries processed:", targetId)


    def verification(self):
        truePositiveDecisions = 0

        for weight, (candidateMatchId, targetId, targetGeom) in self.sorted_list:
            if self.relations.verifyRelations(candidateMatchId, targetId, self.sourceData[candidateMatchId], targetGeom, self.heuristicCondition, self.condition_limit , self.dynamic_factor, self.violation_limit) == 2:
                print("finish the program and return")
                return


    def printResults(self):
        print("\n\nCurrent method", self.getMethodName())
        print("Indexing Time", self.indexingTime)
        print("Initialization Time", self.initializationTime)
        print("Verification Time", self.verificationTime)
        self.relations.print()

    def setThetas(self):
        self.thetaX, self.thetaY = 0, 0
        for sEntity in self.sourceData:
            envelope = sEntity.envelope.bounds
            self.thetaX += envelope[2] - envelope[0]
            self.thetaY += envelope[3] - envelope[1]
        self.thetaX /= len(self.sourceData)
        self.thetaY /= len(self.sourceData)
        print("Dimensions of Equigrid", self.thetaX, "and", self.thetaY)

    def validCandidate(self, candidateId, targetEnv):
        return self.sourceData[candidateId].envelope.intersects(targetEnv)

