

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from PIL import Image
# Root directory of the project
ROOT_DIR = os.path.abspath("../../../")
import matplotlib.pyplot as plt
import cv2
from test import predict_masks

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
from keras import backend as K


class TeethConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "teeth"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class TeethDataset(utils.Dataset):

    def load_teeth(self, dataset_dir, subset):
        # Add classes. We have only one class to add.
        self.add_class("teeth", 1, "teeth")
        points = []
        xrays = []
        counter = 0
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)


        # annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        # annotations = list(annotations.values())  # don't need the dict keys
        # annotations = [a for a in annotations if a['regions']]
        #
        # # #Add images
        # for a in annotations:
        #     if type(a['regions']) is dict:
        #         polygons = [r['shape_attributes'] for r in a['regions'].values()]
        #     else:
        #         polygons = [r['shape_attributes'] for r in a['regions']]
        #
        #     image_path = os.path.join(dataset_dir, a['filename'])
        #     image = skimage.io.imread(image_path)
        #     height, width = image.shape[:2]
        #
        #     self.add_image(
        #         "balloon",
        #         image_id=a['filename'],  # use file name as a unique image id
        #         path=image_path,
        #         width=width, height=height,
        #         polygons=polygons)
        teeth_release_file = 'Labelled json files_labelled xrays_labelled.json'
        for line in open(teeth_release_file, 'r'):
            response = json.loads(line)
            xray = response['image_payload']['image_uri']
            xrays.append(xray)
            landmark_points = []
            polygons = []
            image_path = xray.split('/')
            image_path = image_path[len(image_path) - 1]
            im = Image.open('./dataset/Images/'+image_path)
            width, height = im.size
            for tooth_mark in response['annotations']:
                localpoints = \
                tooth_mark['annotation_value']['image_bounding_poly_annotation']['normalized_bounding_poly'][
                    'normalized_vertices']

                tooth_points = []
                x_points = []
                y_points = []
                print(len(localpoints))
                for point in localpoints:
                    x_coor = point['x']
                    y_coor = point['y']
                    landmark_points.append(point)
                    x_points.append(x_coor * width)
                    y_points.append(y_coor * height)
                    tooth_points.append((x_coor * width, y_coor * height))
                local_teeth = {
                    "name": "polygon",
                    'all_points_x': x_points,
                    'all_points_y': y_points
                }
                polygons.append(local_teeth)

                counter += 1

            self.add_image(
                "teeth",
                image_id='./dataset/Images/'+image_path,  # use file name as a unique image id
                path='./dataset/Images/'+image_path,
                width=width, height=height,
                polygons=polygons)
        a= 35
        print(a)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]


        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1


        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = TeethDataset()
    dataset_train.load_teeth(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = TeethDataset()
    dataset_val.load_teeth(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]


    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        #print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(image_path, as_gray=True)
        #image =cv2.imread(args.image, 1)
        # Detect objects
        image = skimage.color.gray2rgb(image)
        r = model.detect([image], verbose=1)[0]
        mask_perio, mask_cej = predict_masks(image_path)
        #image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        #modellib.load_image_gt(dataset, config, args.image, use_mini_mask=False)
        # utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
        #                        r['rois'], r['class_ids'], r['scores'], r['masks'],
        #                        verbose=1)
        masked_img = visualize.display_instances(
            image,
            r['rois'], r['masks'], r['class_ids'],
            r['class_ids'], mask_perio, mask_cej, image_path)

        # visualize.display_differences(
        #     image,
        #     gt_bbox, gt_class_id, gt_mask,
        #     r['rois'], r['class_ids'], r['scores'], r['masks'],
        #     dataset.class_names, ax=get_ax(),
        #     show_box=False, show_mask=False,
        #     iou_threshold=0.5, score_threshold=0.5)
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output

        file_name = "result"+image_path
        #skimage.io.imsave(file_name, masked_img.astype(np.uint8))
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    #print("Saved to ", file_name)


############################################################
#  Training
############################################################

def driver(image_name):
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    # parser.add_argument("command",
    #                     metavar="<command>",
    #                     help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="Pictures",
                        default="dataset",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=False,
                        default="mask_rcnn_balloon_0039.h5",
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        default='./dataset/Images/9.png',
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    # if args.command == "train":
    #     assert args.dataset, "Argument --dataset is required for training"
    # elif args.command == "splash":
    #     assert args.image or args.video,\
    #            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    config = TeethConfig()
    # if args.command == "train":
    #     config = BalloonConfig()
    # else:
    #     class InferenceConfig(BalloonConfig):
    #         # Set batch size to 1 since we'll be running inference on
    #         # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    #         GPU_COUNT = 1
    #         IMAGES_PER_GPU = 1
    #     config = InferenceConfig()
    config.display()


    class InferenceConfig(TeethConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=args.logs)
    # Create model
    # if args.command == "train":
    #     model = modellib.MaskRCNN(mode="training", config=config,
    #                               model_dir=args.logs)
    # else:
    #     model = modellib.MaskRCNN(mode="inference", config=config,
    #                               model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    detect_and_color_splash(model, image_path=image_name, video_path=args.video)
    #train(model)
    model.keras_model._make_predict_function()

    # if args.command == "train":
    #     train(model)
    # elif args.command == "splash":
    #     detect_and_color_splash(model, image_path=args.image,
    #                             video_path=args.video)
    # else:
    #     print("'{}' is not recognized. "
    #           "Use 'train' or 'splash'".format(args.command))

# image = cv2.imread('4.png')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('4.png', gray)
driver('1.png')
