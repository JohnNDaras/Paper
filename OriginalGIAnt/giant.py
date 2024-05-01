import math
import numpy as np
import random
import sys
import time
from collections import defaultdict
from shapely.geometry import LineString, MultiPolygon, Polygon
from utilities import CsvReader
from datamodel import RelatedGeometries

class GIAnt:

    def __init__(self, qPairs: int, delimiter: str, sourceFilePath: str, targetFilePath: str):
        self.detectedQP = 0
        self.totalCandidatePairs = 0
        self.delimiter = delimiter
        self.sourceData = CsvReader.readAllEntities(delimiter, sourceFilePath)
        self.flag = [-1] * len(self.sourceData)
        self.frequency = [-1] * len(self.sourceData)
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
        self.N = 0

    def applyProcessing(self) :
      time1 = int(time.time() * 1000)
      self.setThetas()
      self.indexSource()
      time2 = int(time.time() * 1000)
      self.verification()
      time3 = int(time.time() * 1000)

      print("Indexing Time\t:\t" + str(time2 - time1))
      print("Verification Time\t:\t" + str(time3 - time2))
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



    def getNoOfBlocks(self, envelope) :
      maxX = math.ceil(envelope[2] / self.thetaX)
      maxY = math.ceil(envelope[3] / self.thetaY)
      minX = math.floor(envelope[0] / self.thetaX)
      minY = math.floor(envelope[1] / self.thetaY)
      return (maxX - minX + 1) * (maxY - minY + 1)


    def verification(self):
        candidate_list = []  # Normal list to store the candidates
        targetId, truePositiveDecisions = 0, 0
        self.relations.reset()
        targetData = CsvReader.readAllEntities(self.delimiter, self.targetFilePath)

        for targetGeom in targetData:
            candidates = self.getCandidates(targetId, targetGeom)
            for candidateMatchId in candidates:
                if self.validCandidate(candidateMatchId, targetGeom.envelope):
                    candidate_list.append((candidateMatchId, targetId, targetGeom))

            targetId += 1

        # Process items from the sorted list
        truePositiveDecisions = 0
        while candidate_list:
            candidateMatchId, targetId, targetGeom = candidate_list.pop(0)  # Process each item
            if self.relations.verifyRelations(candidateMatchId, targetId, self.sourceData[candidateMatchId], targetGeom):
                truePositiveDecisions += 1

        print("True Positive Decisions\t:\t" + str(truePositiveDecisions))
