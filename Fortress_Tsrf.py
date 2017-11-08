import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xmap
from scipy.ndimage.morphology import distance_transform_edt
from rasterio.warp import transform
import seaborn as sns
from scipy import ndimage as nd

sns.set_style('white')
sns.set_context("talk", font_scale=1.2, rc={"lines.linewidth": 2})

# File paths
SD   = r'F:\Work\e\Data\Obs\Canada_Project_Sites\Fortress\Joe_Drone\20170607-SnowPatch1Clip-SDCC10cm.tif'
Tsrf = r'F:\Work\e\Data\Obs\Canada_Project_Sites\Fortress\Joe_Drone\20170607-SnowPatch1Clip-TSurf.tif'
rgb  = r'F:\Work\e\Data\Obs\Canada_Project_Sites\Fortress\Joe_Drone\20170607-SnowPatch1Clip-Ortho.tif'

# Load in
da_SD   = xr.open_rasterio(SD).isel(band=0).drop('band')
da_Tsrf = xr.open_rasterio(Tsrf).isel(band=0).drop('band')
da_rgb  = xr.open_rasterio(rgb)

# rgb to graysacle
da_rgb = da_rgb.mean(dim='band')

# Set missing
da_SD = da_SD.where(da_SD>=0)
da_Tsrf = da_Tsrf.where(da_Tsrf>=-50) # Remove missing values
da_rgb = da_rgb.where(da_rgb>0) # 0 is missing

# Create snow mask from grayscale
from skimage import filters
X = da_rgb.values
thres = filters.threshold_otsu(X[~np.isnan(X)])*1.3 # 20% increase to be more conservative
print(thres)
snow_mask = da_rgb.where(da_rgb>thres)
snow_mask.attrs = da_SD.attrs # add back attributes that get lost above

# Plot check snow_mask
plt.figure()
da_rgb.plot()
plt.figure()
snow_mask.plot()

# Add lat lon coords
# Compute the lon/lat coordinates with rasterio.warp.transform
def add_lat_lon(da_in):
    ny, nx = len(da_in['y']), len(da_in['x'])
    x, y = np.meshgrid(da_in['x'], da_in['y'])

    # Rasterio works with 1D arrays
    lon, lat = transform(da_in.crs, {'init': 'EPSG:4326'},
                         x.flatten(), y.flatten())
    lon = np.asarray(lon).reshape((ny, nx))
    lat = np.asarray(lat).reshape((ny, nx))
    da_in.coords['lon'] = (('y', 'x'), lon)
    da_in.coords['lat'] = (('y', 'x'), lat)

    return da_in

snow_mask = add_lat_lon(snow_mask)
da_SD = add_lat_lon(da_SD)
da_Tsrf = add_lat_lon(da_Tsrf)

# Fill in missing SD
def fill(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. 
                 data value are replaced where invalid is True
                 If None (default), use: invalid  = np.isnan(data)

    Output: 
        Return a filled array. 
    """
    if invalid is None: invalid = np.isnan(data)

    ind = nd.distance_transform_edt(invalid,
                                    return_distances=False,
                                    return_indices=True)
    return data[tuple(ind)]
da_SD.data = fill(da_SD.data)

# grids are same extent but SD is 10cm and Tsrf is ~25cm
# Down-sample Tsrf to be snow_mask resolution
da_Tsrf.xmap.set_coords('lon', 'lat')
da_Tsrf_regrid = da_Tsrf.xmap.remap_like(snow_mask, xcoord='lon', ycoord='lat', tcoord=None, how='nearest', k=1) #.where(da_SD.notnull())
da_Tsrf_regrid.name = da_Tsrf.name

# Mask to snowcovered areas
da_Tsrf_regrid = da_Tsrf_regrid.where(snow_mask.notnull())

# Mask Tsrf to temps below 8
da_Tsrf_regrid = da_Tsrf_regrid.where(da_Tsrf_regrid<8)

# Per snow pixel, find nearest distance to edge
pixel_dist = 0.1 # m
a_d = distance_transform_edt(da_Tsrf_regrid.notnull().values) # fill missing to zero
da_d = da_Tsrf_regrid.copy(deep=True)
da_d.data = a_d * pixel_dist # Convert from pixel units to meters

# Plot check regridding
plt.figure()
da_SD.plot()

plt.figure()
da_Tsrf.plot()

plt.figure()
da_Tsrf_regrid.plot()

plt.figure()
da_d.plot()

# Plot Tsrf vs. distance to edge
plt.figure()
plt.scatter(da_d.values[:], da_Tsrf_regrid.values[:])

# 2d hist plot
I_ok = (da_d.notnull()) & (da_Tsrf_regrid.notnull())
x = da_d.where(I_ok, drop=True).values.flatten()
y = da_Tsrf_regrid.where(I_ok, drop=True).values.flatten()
hexplot = sns.jointplot(y=y, x=x, kind='hex', color='k',
                  gridsize=20).set_axis_labels('Distance from snowpatch edge (m)','Brightness Temperature (C)')

plt.show()

