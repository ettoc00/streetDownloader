# streetdownloader

## Introduction
streetdownloader is a Python package that uses the 
Google Street View API to download panoramic images. 
It allows the user to download both full panoramic images via scraping 
and various parts of the panorama in perspective distortion via the API.

## Requirements

To run this package, you will need to create a `.env` file 
and insert a `GOOGLE_API_KEY` obtained through the Google Cloud Platform 
at https://developers.google.com/maps/documentation/streetview/get-api-key

Please note that downloading panoramic images is free, 
while downloading images in perspective distortion (views) 
may incur a cost of $0.10 per location, as 14 images are downloaded per location. 
More information can be found at https://developers.google.com/maps/documentation/streetview/usage-and-billing

This package was developed and tested using Python 3.10, 
but it should be compatible with Python 3.9 and later versions.

The package's dependencies can be installed using either 
`pip install -r requirements.txt` or `poetry install`.


## Usage
The package has the following functions:

### download_panoramas
`download_panoramas(folder: StrOrBytesPath, loc1: Location, loc2: Location)`

This function downloads all the panoramic images 
in the area defined by the two locations provided. 
The images will be saved to the specified path 
and named with the id of the panorama. 

Location is a named tuple containing latitude and longitude. 
A tuple of two floats can also be passed as an argument

### download_views
`download_views(folder: StrOrBytesPath, loc1: Location, loc2: Location)`

This function downloads all the images in perspective distortion 
in the area defined by the two locations provided. 
The images will be saved to the specified path 
and named with a name that includes the panorama's ID and information
about the viewpoint, such as heading, field of view (fov), and pitch.

### user_input
`user_input()`

This function prompts the user for input, and returns a tuple 
containing the path where the images will be saved, 
and the starting and ending locations of the area where the 
images should be downloaded.

Please note that to select the folder you need to have the tkinter library installed. 
To prompt the location, you need to have Google Chrome and chromedriver installed. 
If chromedriver is not found, the library `chromedriver-autoinstaller` will try to install it automatically.

## Example
Using `user_input`:
```python
from streetdownloader import download_panoramas, user_input

folder, loc1, loc2 = user_input()
download_panoramas(folder, loc1, loc2)
```

Manual input:
```python
from streetdownloader import download_panoramas, Location, Path

folder = Path('/path/to/colosseum')
loc1 = 41.8914369,12.4907346
loc2 = Location(41.8890246,12.4936502)
download_panoramas(folder, loc1, loc2)
```


## Note
Please be aware that the usage of the Google Street View API is subject
to their terms and conditions, 
and that this package is for research purposes only.