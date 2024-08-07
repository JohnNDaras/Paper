from shapely import relate
from shapely import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon
from shapely import get_num_coordinates
from de9im_patterns import contains, crosses_lines, crosses_1, crosses_2, disjoint, equal, intersects, overlaps1, overlaps2, touches, within, covered_by, covers

class RelatedGeometries :
        def __init__(self, qualifyingPairs) :
            self.pgr = 0
            self.exceptions = 0
            self.detectedLinks = 0
            self.verifiedPairs = 0
            self.qualifyingPairs = qualifyingPairs
            self.interlinkedGeometries = 0
            self.continuous_unrelated_Pairs = 0
            self.violations = 0
            self.containsD1 = []
            self.containsD2 = []
            self.coveredByD1 = []
            self.coveredByD2 = []
            self.coversD1 = []
            self.coversD2 = []
            self.crossesD1 = []
            self.crossesD2 = []
            self.equalsD1 = []
            self.equalsD2 = []
            self.intersectsD1 = []
            self.intersectsD2 = []
            self.overlapsD1 = []
            self.overlapsD2 = []
            self.touchesD1 = []
            self.touchesD2 = []
            self.withinD1 = []
            self.withinD2 = []

        def addContains(self, gId1,  gId2) :
          self.containsD1.append(gId1)
          self.containsD2.append(gId2)
        def addCoveredBy(self, gId1,  gId2):
           self.coveredByD1.append(gId1)
           self.coveredByD2.append(gId2)
        def addCovers(self, gId1,  gId2):
           self.coversD1.append(gId1)
           self.coversD2.append(gId2)
        def addCrosses(self, gId1,  gId2) :
          self.crossesD1.append(gId1)
          self.crossesD2.append(gId2)
        def addEquals(self, gId1,  gId2) :
          self.equalsD1.append(gId1)
          self.equalsD2.append(gId2)
        def addIntersects(self, gId1,  gId2) :
          self.intersectsD1.append(gId1)
          self.intersectsD2.append(gId2)
        def addOverlaps(self, gId1,  gId2) :
          self.overlapsD1.append(gId1)
          self.overlapsD2.append(gId2)
        def addTouches(self, gId1,  gId2) :
          self.touchesD1.append(gId1)
          self.touchesD2.append(gId2)
        def addWithin(self, gId1,  gId2) :
          self.withinD1.append(gId1)
          self.withinD2.append(gId2)

        def  getInterlinkedPairs(self) :
            return self.interlinkedGeometries
        def  getNoOfContains(self) :
            return len(self.containsD1)
        def  getNoOfCoveredBy(self) :
            return len(self.coveredByD1)
        def  getNoOfCovers(self) :
            return len(self.coversD1)
        def  getNoOfCrosses(self) :
            return len(self.crossesD1)
        def  getNoOfEquals(self) :
            return len(self.equalsD1)
        def  getNoOfIntersects(self) :
            return len(self.intersectsD1)
        def  getNoOfOverlaps(self) :
            return len(self.overlapsD1)
        def  getNoOfTouches(self) :
            return len(self.touchesD1)
        def  getNoOfWithin(self) :
            return len(self.withinD1)
        def  getVerifiedPairs(self) :
            return self.verifiedPairs

        def reset(self):
            self.pgr = 0
            self.exceptions = 0
            self.detectedLinks = 0
            self.verifiedPairs = 0
            self.interlinkedGeometries = 0

            self.containsD1.clear()
            self.containsD2.clear()
            self.coveredByD1.clear()
            self.coveredByD2.clear()
            self.coversD1.clear()
            self.coversD2.clear()
            self.crossesD1.clear()
            self.crossesD2.clear()
            self.equalsD1.clear()
            self.equalsD2.clear()
            self.intersectsD1.clear()
            self.intersectsD2.clear()
            self.overlapsD1.clear()
            self.overlapsD2.clear()
            self.touchesD1.clear()
            self.touchesD2.clear()
            self.withinD1.clear()
            self.withinD2.clear()


        def print(self) :
            print("Qualifying pairs:\t", str(self.qualifyingPairs))
            print("Exceptions:\t", str(self.exceptions))
            print("Detected Links:\t", str(self.detectedLinks))
            print("Interlinked geometries:\t", str(self.interlinkedGeometries))
            print("No of contains:\t", str(self.getNoOfContains()))
            print("No of covered-by:\t" + str(self.getNoOfCoveredBy()))
            print("No of covers:\t", str(self.getNoOfCovers()))
            print("No of crosses:\t", str(self.getNoOfCrosses()))
            print("No of equals:\t", str(self.getNoOfEquals()))
            print("No of intersects:\t" + str(self.getNoOfIntersects()))
            print("No of overlaps:\t", str(self.getNoOfOverlaps()))
            print("No of touches:\t", str(self.getNoOfTouches()))
            print("No of within:\t", str(self.getNoOfWithin()))

            if self.qualifyingPairs != 0:
              print("Recall", str((self.interlinkedGeometries / float(self.qualifyingPairs))))
            else:
              print('array is empty')
            if self.verifiedPairs != 0:
              print("Precision", str((self.interlinkedGeometries / self.verifiedPairs)))
            else:
              print('array is empty 2')
            if self.qualifyingPairs != 0 and self.verifiedPairs != 0:
              print("Progressive Geometry Recall", str(self.pgr / self.qualifyingPairs / self.verifiedPairs))
            else:
              print('array is empty 3')
            print("Verified pairs", str(self.verifiedPairs))


        def Precision_Threshold(self, ConditionLimit):
          if self.verifiedPairs != 0:
              Precision = self.interlinkedGeometries / float(self.verifiedPairs)
              if Precision <= ConditionLimit:
                 return 1


        def Qualifying_Distance_Threshold(self, ConditionLimit):
          if self.continuous_unrelated_Pairs  >= ConditionLimit:
            print("Continuous unrelated pairs: ",self.continuous_unrelated_Pairs)
            return 1


        def Dynamic_Qualifying_Distance_Threshold(self,ConditionLimit):
          if self.continuous_unrelated_Pairs >= ConditionLimit:
            return 1




        # Function to determine the dimension based on geometry type
        def get_dimension(self,geometry):
            if isinstance(geometry, Point) or geometry.geom_type == 'Point' or geometry.geom_type == 'MultiPoint':
                return 0
            elif isinstance(geometry, LineString) or geometry.geom_type == 'LineString' or geometry.geom_type == 'LinearRing' or geometry.geom_type == 'MultiLineString':
                return 1
            elif isinstance(geometry, Polygon) or geometry.geom_type == 'Polygon' or geometry.geom_type == 'MultiPolygon':
                return 2
            else:
                return None

        # Check if two geometries have the same dimension
        def have_same_dimension(self,geom1, geom2):
            return self.get_dimension(geom1) == self.get_dimension(geom2)

        def  verifyRelations(self, geomId1,  geomId2,  sourceGeom,  targetGeom, heuristicCondition, ConditionLimit, Dynamic_Factor, violation_limit) :
            related = False
            self.verifiedPairs += 1
            array = relate(sourceGeom, targetGeom)

            if intersects.matches(array):
                related = True
                self.detectedLinks += 1
                self.addIntersects(geomId1, geomId2)
            if within.matches(array):
                related = True
                self.detectedLinks += 1
                self.addWithin(geomId1, geomId2)
            if covered_by.matches(array):
                related = True
                self.detectedLinks += 1
                self.addCoveredBy(geomId1, geomId2)
            if self.get_dimension(sourceGeom) == self.get_dimension(targetGeom) ==1:
              if crosses_lines.matches(array):
                related = True
                self.detectedLinks += 1
                self.addCrosses(geomId1, geomId2)
            elif self.get_dimension(sourceGeom) > self.get_dimension(targetGeom) :
              if crosses_2.matches(array):
                related = True
                self.detectedLinks += 1
                self.addCrosses(geomId1, geomId2)
            elif self.get_dimension(sourceGeom) < self.get_dimension(targetGeom) :
              if crosses_1.matches(array):
                related = True
                self.detectedLinks += 1
                self.addCrosses(geomId1, geomId2)
            if self.have_same_dimension(sourceGeom, targetGeom):
              if overlaps1.matches(array) or overlaps2.matches(array):
                  related = True
                  self.detectedLinks += 1
                  #print(sourceGeom,"  ",targetGeom)
                  self.addOverlaps(geomId1, geomId2)
            if  equal.matches(array):
                related = True
                self.detectedLinks += 1
                self.addEquals(geomId1, geomId2)
            if  touches.matches(array):
                related = True
                self.detectedLinks += 1
                self.addTouches(geomId1, geomId2)
            if  contains.matches(array):
                related = True
                self.detectedLinks += 1
                self.addContains(geomId1, geomId2)
            if covers.matches(array):
                related = True
                self.detectedLinks += 1
                self.addCovers(geomId1, geomId2)

            if (related) :
                self.interlinkedGeometries += 1
                self.pgr += self.interlinkedGeometries
                self.continuous_unrelated_Pairs = 0
            else:
                self.continuous_unrelated_Pairs += 1


            if violation_limit == 0:
              if heuristicCondition == "Precision_Threshold":
                if self.Precision_Threshold(ConditionLimit) == 1:
                      return 2
              if heuristicCondition == "Qualifying_Distance_Threshold":
                if self.Qualifying_Distance_Threshold(ConditionLimit) == 1:
                      return 2
              if heuristicCondition == "Dynamic_Qualifying_Distance_Threshold":
                ConditionLimit = ConditionLimit + Dynamic_Factor
                if self.Dynamic_Qualifying_Distance_Threshold(ConditionLimit) == 1:
                  return 2

            if violation_limit != 0:     #buffered threshold case
                if heuristicCondition == "Precision_Threshold":
                  if self.Precision_Threshold(ConditionLimit) == 1:
                    if self.violations < violation_limit:
                        self.violations += 1
                    else:
                        return 2
                if heuristicCondition == "Qualifying_Distance_Threshold":
                  if self.Qualifying_Distance_Threshold(ConditionLimit) == 1:
                    if self.violations < violation_limit:
                        self.violations += 1
                    else:
                        return 2
                if heuristicCondition == "Dynamic_Qualifying_Distance_Threshold":
                  ConditionLimit = ConditionLimit + Dynamic_Factor
                  if self.Dynamic_Qualifying_Distance_Threshold(ConditionLimit) == 1:
                    if self.violations < violation_limit:
                        self.violations += 1
                    else:
                        return 2

            return related
