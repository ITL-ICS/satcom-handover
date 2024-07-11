import random
import pyproj

class GeoCordinates:
    def __init__(self, lat, lon, alt):
        self.lat = lat
        self.lon = lon
        self.alt = alt

    def geo2ecef(self):
        # Define the WGS84 ellipsoid
        geod = pyproj.CRS('EPSG:4326')
        
        # Define the ECEF coordinate system
        ecef = pyproj.CRS('EPSG:4978')
        
        # Define the pyproj transformer for geodetic to ECEF transformation
        transformer = pyproj.Transformer.from_crs(geod, ecef)
        
        return transformer.transform(self.lat, self.lon, self.alt)

"""coordinates = GeoCordinates(37.5665, 126.9780, 0)    
print("lat의 타입:", type(coordinates.lat))
ecef_coordinates = coordinates.geo2ecef()

print("ECEF 좌표:", ecef_coordinates)
print("ECEF의 타입:", type(ecef_coordinates))"""
