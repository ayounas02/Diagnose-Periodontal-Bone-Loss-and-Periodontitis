
import json
import ast
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from math import atan2, sin, cos, sqrt, pi, degrees
import matplotlib.pyplot as plt
import shapely
from shapely import geometry
def area(pts):
    'Area of cross-section.'

    if pts[0] != pts[-1]:
        pts = pts + pts[:1]
    x = [c[0] for c in pts]
    y = [c[1] for c in pts]
    s = 0
    for i in range(len(pts) - 1):
        s += x[i] * y[i + 1] - x[i + 1] * y[i]
    return s / 2


def centroid(pts):
    'Location of centroid.'

    if pts[0] != pts[-1]:
        pts = pts + pts[:1]
    x = [c[0] for c in pts]
    y = [c[1] for c in pts]
    sx = sy = 0
    a = area(pts)
    for i in range(len(pts) - 1):
        sx += (x[i] + x[i + 1]) * (x[i] * y[i + 1] - x[i + 1] * y[i])
        sy += (y[i] + y[i + 1]) * (x[i] * y[i + 1] - x[i + 1] * y[i])
    return sx / (6 * a), sy / (6 * a)


def inertia(pts):
    'Moments and product of inertia about centroid.'

    if pts[0] != pts[-1]:
        pts = pts + pts[:1]
    x = [c[0] for c in pts]
    y = [c[1] for c in pts]
    sxx = syy = sxy = 0
    a = area(pts)
    cx, cy = centroid(pts)
    for i in range(len(pts) - 1):
        sxx += (y[i] ** 2 + y[i] * y[i + 1] + y[i + 1] ** 2) * (x[i] * y[i + 1] - x[i + 1] * y[i])
        syy += (x[i] ** 2 + x[i] * x[i + 1] + x[i + 1] ** 2) * (x[i] * y[i + 1] - x[i + 1] * y[i])
        sxy += (x[i] * y[i + 1] + 2 * x[i] * y[i] + 2 * x[i + 1] * y[i + 1] + x[i + 1] * y[i]) * (
                    x[i] * y[i + 1] - x[i + 1] * y[i])
    return sxx / 12 - a * cy ** 2, syy / 12 - a * cx ** 2, sxy / 24 - a * cx * cy


def principal(Ixx, Iyy, Ixy):
    'Principal moments of inertia and orientation.'

    avg = (Ixx + Iyy) / 2
    diff = (Ixx - Iyy) / 2  # signed
    I1 = avg + sqrt(diff ** 2 + Ixy ** 2)
    I2 = avg - sqrt(diff ** 2 + Ixy ** 2)
    theta = atan2(-Ixy, diff) / 2
    return I1, I2, theta


def outline(pts, basename='section', format='pdf', size=(8, 8), dpi=100):
    'Draw an outline of the cross-section with centroid and principal axes.'

    if pts[0] != pts[-1]:
        pts = pts + pts[:1]
    x = [c[0] for c in pts]
    y = [c[1] for c in pts]

    # Get the bounds of the cross-section
    minx = min(x)
    maxx = max(x)
    miny = min(y)
    maxy = max(y)

    # Whitespace border is 5% of the larger dimension
    b = .05 * max(maxx - minx, maxy - miny)

    # Get the properties needed for the centroid and principal axes
    cx, cy = centroid(pts)
    i = inertia(pts)
    p = principal(*i)


    # Principal axes extend 10% of the minimum dimension from the centroid
    length = min(maxx - minx, maxy - miny)
    #length=length/
    a1x = [cx - length * cos(p[2]), cx + length * cos(p[2])]
    a1y = [cy - length * sin(p[2]), cy + length * sin(p[2])]
    a2x = [cx - length * cos(p[2] + pi / 2), cx + length * cos(p[2] + pi / 2)]
    a2y = [cy - length * sin(p[2] + pi / 2), cy + length * sin(p[2] + pi / 2)]

    if abs(a2y[0] - a2y[1]) > abs(a1y[0] - a1y[1]):
        return a2x, a2y, cx, cy, b
    else:
        return a1x, a1y, cx, cy, b


    # Plot and save
    # Axis colors chosen from http://mkweb.bcgsc.ca/colorblind/
    fig, ax = plt.subplots(figsize=size)
    ax.plot(x, y, 'k*-', lw=2)
    #plt.savefig('1.png', format=format, dpi=dpi)
    ax.plot(a1x, a1y, '-', color='#0072B2', lw=2)  # blue
    #plt.savefig('2.png', format=format, dpi=dpi)
    ax.plot(a2x, a2y, '-', color='#D55E00')  # vermillion
    #plt.savefig('3.png', format=format, dpi=dpi)
    ax.plot(cx, cy, 'ko', mec='k')
    #plt.savefig('4.png', format=format, dpi=dpi)
    ax.set_aspect('equal')
    plt.xlim(xmin=minx - b, xmax=maxx + b)
    #plt.savefig('5.png', format=format, dpi=dpi)
    plt.ylim(ymin=miny - b, ymax=maxy + b)
    #plt.savefig('6.png', format=format, dpi=dpi)
    filename = basename + '.' + format
    #plt.savefig(filename, format=format, dpi=dpi)
    # plt.show()
    # plt.close()




def plot_points(points, img_list):
    for idx, img in enumerate(img_list):
        image_path = img.split('/')
        image_path = image_path[len(image_path)-1]
        im = Image.open(image_path)
        width, height = im.size
        im = plt.imread(image_path)

        for coordinate in points:
            xs = []
            ys = []
            for i in range(len(coordinate)):
                x = coordinate[i]['x'] * width
                y = coordinate[i]['y'] * height
                xs.append(x)
                ys.append(y)
            # Below we are using data attribute
            plt.fill("j", "k", 'm',
                     data={"j":xs,
                           "k": ys})  # here 'm' for magenta
            plt.savefig('ss6.png', format=format)

            implot = plt.imshow(im)
            # put a blue dot at (10, 20)
            # plt.scatter(x, y)

            # put a red dot, size 40, at 2 locations:
            plt.scatter(x=xs, y=ys, c='r', s=2)
            plt.show()





def find_intersection(line_coordinates, teeth_coordinates):

    polygon = [(4.0, -2.0), (5.0, -2.0), (4.0, -3.0), (3.0, -3.0), (4.0, -2.0)]
    polygon = teeth_coordinates
    shapely_poly = shapely.geometry.Polygon(polygon)

    line = [(4.0, -2.0000000000000004), (2.0, -1.1102230246251565e-15)]
    line = line_coordinates
    shapely_line = shapely.geometry.LineString(line)
    try:
        intersection_line = list(shapely_poly.intersection(shapely_line).coords)
        return intersection_line
    except:
        return []


    #print(intersection_line)









def main():
    points = []
    xrays = []
    counter = 0
    for line in open('./dataset/Labelled json files_labelled xrays_labelled.json', 'r'):
        response = json.loads(line)
        xray = response['image_payload']['image_uri']
        xrays.append(xray)
        landmark_points = []

        for tooth_mark in response['annotations']:
            localpoints = tooth_mark['annotation_value']['image_bounding_poly_annotation']['normalized_bounding_poly']['normalized_vertices']
            image_path = xray.split('/')
            image_path = image_path[len(image_path) - 1]
            im = Image.open(image_path)
            width, height = im.size
            tooth_points = []
            x_points = []
            y_points = []
            for point in localpoints:
                x_coor = point['x']
                y_coor = point['y']
                landmark_points.append(point)
                x_points.append(x_coor *width)
                y_points.append(y_coor*height)
                tooth_points.append((x_coor *width,y_coor*height ))
            plt.fill("j", "k", 'm',
                     data={"j": x_points,
                           "k": y_points})  # here 'm' for magenta
            plt.show()
            plt.savefig(str(counter) + '.png' )
            outline(tooth_points, 'skewed'+str(counter), format='png', size=(8, 6))
            counter += 1

        points.append(landmark_points)



    plot_points(points, xrays)

    #process_data('./dataset/Labelled json files_labelled xrays_labelled.json')

    from segments import SegmentsDataset
    from segments.utils import export_dataset

    # Initialize a SegmentsDataset from the release file
    release_file = './dataset/Labelled json files_Perio1-v0.1.json'
    dataset = SegmentsDataset(release_file, labelset='ground-truth', filter_by=['labeled'])

    #export_dataset(dataset, export_format='coco-panoptic')
    import matplotlib.pyplot as plt
    from segments.utils import get_semantic_bitmap

    for sample in dataset:
        # Print the sample name and list of labeled objects
        print(sample['name'])
        print(sample['annotations'])

        # Show the image
        #plt.imshow(sample['image'])
        #plt.show()

        # Show the instance segmentation label
        #plt.imshow(sample['segmentation_bitmap'])
        #plt.show()

        # Show the semantic segmentation label
        semantic_bitmap = get_semantic_bitmap(sample['segmentation_bitmap'], sample['annotations'])
        #cv2.imwrite('mask_'+sample['name'], semantic_bitmap)
        plt.imsave('mask_'+sample['name'], semantic_bitmap)
        #plt.imshow(semantic_bitmap)
        #plt.show()

