from geopy.geocoders import Nominatim

def get_lat_long(location_name):
    geolocator = Nominatim(user_agent="geo_locator")
    location = geolocator.geocode(location_name)
    if location:
        latitude = location.latitude
        longitude = location.longitude
        return latitude, longitude
    else:
        return None, None


