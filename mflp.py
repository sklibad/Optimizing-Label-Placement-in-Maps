import geopandas as gpd
from additional_functions import *
from rasterio import open as rio_open
from pyproj import CRS
import geopandas as gpd
from shapely.ops import unary_union

settlements_path = "sidla.shp"
collision_path = "Z_Voda_P.shp"
raster_path = "ztm25_benesovsko.tif"
output_path = "benesovsko_labels.shp"
font_size = None
font_color_rgb = [0, 0, 0]
M = 25000

def main():
    with rio_open(raster_path) as src:
        crs = CRS.from_user_input(src.crs)

    gdf_settlements = gpd.read_file(settlements_path).to_crs(crs)
    collision_gdf = unary_union(gpd.read_file(collision_path).to_crs(crs).geometry)

    offset_distance = 0.0005*M
    
    gdf_labels = generate_label_candidates(gdf_settlements, M, font_size)

    final_labels = get_final_label_positions(gdf_settlements, gdf_labels, offset_distance, offset_distance/2, collision_gdf)

    map_load_raster, contrast_raster = get_contrast_and_map_load(raster_path, text_color_rgb=font_color_rgb)
    
    result_gdf = find_the_best_positions(contrast_raster, map_load_raster, final_labels)
    result_gdf = result_gdf[["dist", "dir_prio", "map_load", "contrast", "geometry"]]

    result_gdf.to_file(output_path)

if __name__ == "__main__":
    main()