import radarcoverage.geometry as geom
from radarcoverage import file_utils


if __name__ == '__main__':

    file_utils.chdir_to_file_dir(__file__)

    # 1. Load data

    place = geom.generate_place_from_rooftops_file('../data/small.geojson')

    place.show_sharp_edges_animation()
