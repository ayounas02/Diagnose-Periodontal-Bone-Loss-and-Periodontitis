import matplotlib.pyplot as plt
from segments.utils import get_semantic_bitmap
from segments import SegmentsDataset
from constants import *
import json
from PIL import Image
from matplotlib.patches import Rectangle
import os
import cv2
b
def create_folders(path):
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)

def plot_points(points, img_list):
    for idx, img in enumerate(img_list):
        image_path = img.split('/')
        image_path = image_path[len(image_path)-1]
        image_path = './dataset/Images/'+image_path
        im = Image.open(image_path)
        width, height = im.size
        im = plt.imread(image_path)
        #all coordinates
        xs = []
        ys = []
        for coordinate in points[idx]:
            x = coordinate['x'] * width
            y = coordinate['y'] * height
            xs.append(x)
            ys.append(y)

            # Below we are using data attribute
                # put a red dot, size 40, at 2 locations:
        plt.scatter(x=xs, y=ys, c='r', s=2)
        fig = plt.imshow(im)
        # plt.imssave(img)
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.savefig('asda.png', bbox_inches='tight', pad_inches=0)
        plt.show()


def read_landmarks():
    points = []
    xrays = []
    counter = 0
    for line in open(teeth_release_file, 'r'):
        response = json.loads(line)
        xray = response['image_payload']['image_uri']
        xrays.append(xray)
        landmark_points = []
        polygons = []
        image_path = xray.split('/')
        image_path = image_path[len(image_path) - 1]
        print(image_path)
        im = Image.open('./dataset/Images/'+image_path)
        width, height = im.size
        x_points = []
        y_points = []
        for tooth_mark in response['annotations']:
            localpoints = tooth_mark['annotation_value']['image_bounding_poly_annotation']['normalized_bounding_poly'][
                'normalized_vertices']
            tooth_points = []
            x_instance_points = []
            y_instance_points = []
            #print(len(localpoints))
            for point in localpoints:
                x_coor = point['x']
                y_coor = point['y']
                landmark_points.append(point)
                x_points.append(x_coor * width)
                x_instance_points.append(x_coor * width)
                y_instance_points.append(y_coor * height)
                y_points.append(y_coor * height)
                tooth_points.append((x_coor * width, y_coor * height))
            local_teeth = {
                "name": "polygon",
                'all_points_x': x_instance_points,
                'all_points_y': y_instance_points
            }
            polygons.append(local_teeth)
            #plt.imshow(Image.open('stinkbug.png'))
            #plt.gca().add_patch(Rectangle((minx, miny), ((minx+maxx)/2-minx)*2, ((miny+maxy)/2-miny)*2, linewidth=1, edgecolor='r',facecolor='none'))

            # plt.fill("j", "k", 'm',
            #          data={"j": x_points,
            #                "k": y_points})



            #outline(tooth_points, 'skewed' + str(counter), format='png', size=(8, 6))
            counter += 1
        plt.savefig('test.png')
        #plt.show()
        points.append(landmark_points)
    plot_points(points, xrays)


def extract_masks( release_file ):
    dataset = SegmentsDataset(release_file, labelset='ground-truth', filter_by=['labeled'])
    path = dataset.image_dir
    if not os.path.exists(path + '/masks'):
        os.makedirs(path + '/masks/')
    for sample in dataset:
        # Show the semantic segmentation label
        semantic_bitmap = get_semantic_bitmap(sample['segmentation_bitmap'], sample['annotations'])
        plt.imsave(path + '/masks/' + sample['name'], semantic_bitmap)

        img = cv2.imread(path + '/masks/' + sample['name'], cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(path + '/masks/' + sample['name'], img)


create_folders('result')
create_folders("../teeth/result/perio/")
create_folders("../teeth/result/cej/")

create_folders('checkpoints')
create_folders("../teeth/checkpoints/perio/")
create_folders("../teeth/checkpoints/cej/")
# Extract Perio
extract_masks(perio_release_file )

# Extract CEJs
extract_masks( cej_release_file )

###### Extract teeth
# Plot Points
#read_landmarks()


create_folders('result')
create_folders("../teeth/result/perio/")
# create_folders(path)
# create_folders(path)


#plot_points(points, xrays)

# Fill region


# save




