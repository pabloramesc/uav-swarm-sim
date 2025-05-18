import rasterio
from rasterio.transform import from_bounds
import contextily as ctx
from pyproj import Transformer
import imageio

with rasterio.open("data/elevation/barcelona_dem.tif") as dem:
    print("Bounds:", dem.bounds)
    print("CRS:", dem.crs)


transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
left, bottom = transformer.transform(dem.bounds[0], dem.bounds[1])
right, top = transformer.transform(dem.bounds[2], dem.bounds[3])
bounds_3857 = (left, bottom, right, top)

print("Bounds in EPSG:3857:", bounds_3857)

try:
    img, ext = ctx.bounds2img(*bounds_3857, zoom=14, source=ctx.providers.Esri.WorldImagery)
    print("Satellite image downloaded successfully")
except Exception as e:
    print("Error downloading tiles:", e)

print("Returned extent (EPSG:3857):", ext)

imageio.imwrite("barcelona_satellite.png", img)

print("Saved satellite image as 'barcelona_satellite.png'")

from PIL import Image

xmin_img, xmax_img, ymin_img, ymax_img = ext

# Image shape
height, width, _ = img.shape

# Calculate pixel sizes
pixel_size_x = (xmax_img - xmin_img) / width
pixel_size_y = (ymax_img - ymin_img) / height

# Calculate pixel coordinates to crop
left_px = int((bounds_3857[0] - xmin_img) / pixel_size_x)
right_px = int((bounds_3857[2] - xmin_img) / pixel_size_x)
top_px = int((ymax_img - bounds_3857[3]) / pixel_size_y)
bottom_px = int((ymax_img - bounds_3857[1]) / pixel_size_y)

if left_px < 0 or right_px > width or top_px < 0 or bottom_px > height:
    raise ValueError("Calculated crop bounds fall outside the image dimensions.")

print(
    f"Cropping pixels (left, right, top, bottom): {left_px}, {right_px}, {top_px}, {bottom_px}"
)

# Crop image (remember numpy array is [row, col] = [y, x])
cropped_img = img[top_px:bottom_px, left_px:right_px]

# Convert to PIL Image and save
cropped_pil = Image.fromarray(cropped_img)
cropped_pil.save("barcelona_satellite_cropped.png")
