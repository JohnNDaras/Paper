import csv
from shapely.geometry import shape
from shapely.wkt import loads
import sys

maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

class CsvReader:
    @staticmethod
    def readAllEntities(delimiter, inputFilePath):
        loadedEntities = []
        geoCollections = 0
        lineCount = 0  # Counter for lines read

        with open(inputFilePath, newline='') as f:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                found_geometry = False
                for column in row:
                    try:
                        geometry = shape(loads(column))
                        found_geometry = True
                        break
                    except Exception as e:
                        print(f"Error parsing column as geometry: {e}")
                if not found_geometry:
                    print("No valid geometry found in row")
                    continue

                if geometry.geom_type == "GeometryCollection":
                    geoCollections += 1
                else:
                    loadedEntities.append(geometry)
        return loadedEntities
