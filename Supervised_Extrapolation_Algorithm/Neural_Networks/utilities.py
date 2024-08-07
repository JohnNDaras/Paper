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
        empty_geometries = 0
        lineCount = 0  # Counter for lines read
        geometry_column_index = None  # Will hold the index of the column with valid geometries

        with open(inputFilePath, newline='') as f:
            reader = csv.reader(f, delimiter=delimiter)
            first_row = True
            for row in reader:
                if first_row:
                    # Check each column in the first row to find the one with a valid geometry
                    for index, column in enumerate(row):
                        try:
                            shape(loads(column))  # Try to create a shape
                            geometry_column_index = index
                            break  # Exit the loop once a valid geometry is found
                        except:
                            continue
                    first_row = False
                    if geometry_column_index is None:
                        print("No valid geometry column found")
                        return []  # Return an empty list if no valid geometry column found

                try:
                    geometry = shape(loads(row[geometry_column_index]))
                except Exception as e:
                    print(f"Error parsing geometry from column {geometry_column_index}: {e}")
                    continue

                if geometry.geom_type == "GeometryCollection":
                    geoCollections += 1
                elif geometry.is_empty or not geometry.is_valid:
                    empty_geometries += 1
                else:
                    lineCount += 1
                    loadedEntities.append(geometry)
        #print(loadedEntities)
        print("Empty geometries count:", empty_geometries)
        return loadedEntities
