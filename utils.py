"""
The utility functions used in FGCN
"""

import sys
import os
import math
import random
import numpy as np
import tensorflow as tf
import scipy
import skimage.color
import skimage.io
import skimage.transform
import urllib.request
import shutil
import warnings
import keras.backend as K
from keras.utils import get_file

############################################################
#  Bounding Boxes
############################################################
def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


############################################################
#  Dataset
############################################################
class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    def append_data(self, class_info, image_info):
        self.external_to_class_id = {}
        for i, c in enumerate(self.class_info):
            for ds, id in c["map"]:
                self.external_to_class_id[ds + str(id)] = i

        # Map external image IDs to internal ones.
        self.external_to_image_id = {}
        for i, info in enumerate(self.image_info):
            self.external_to_image_id[info["ds"] + str(info["id"])] = i

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's availble online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


def resize_image(image, bb3d, bb2d, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged and output the resized labels for BoxCars116k.
    bb3d: [8, 2] The eight 3D corners' coordinates in the image plane. Please note that in some samples, the coordinate
    of some corners may be negtive numbers. It means that these corners fall outside the image.

    bb2d: [x_topleft, y_topleft, width, height] The 2D bounding box provided by the BoxCars116k. Please note that the
    form of this bb2d. it is not the same form as the form used in MaskRCNN, i.e., not [y1, x1, y2, x2]

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = skimage.transform.resize(
            image, (round(h * scale), round(w * scale)),
            order=1, mode="constant", preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
        resized_bb2d = scale * bb2d + np.array([left_pad, top_pad, 0, 0])
        resized_bb3d = scale * bb3d + np.tile(np.array([left_pad, top_pad]), (8, 1))

        y1 = resized_bb2d[1]
        x1 = resized_bb2d[0]
        y2 = resized_bb2d[1] + resized_bb2d[3]
        x2 = resized_bb2d[0] + resized_bb2d[2]

        resized_bb2d = np.zeros((1, 4), dtype=np.float32)
        resized_bb2d[0, 0] = y1
        resized_bb2d[0, 1] = x1
        resized_bb2d[0, 2] = y2
        resized_bb2d[0, 3] = x2

    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), resized_bb3d, resized_bb2d, window, scale, padding, crop


############################################################
#  Anchors
############################################################
def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # In retinanet, there are 9 anchors per location rather than 3 anchors in original
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    scale_factor = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    for i in range(len(scales)):
        anchors.append(generate_anchors(scale_factor*scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)


############################################################
#  Homography implementation by tensorflow
############################################################
def ax(p_x, p_y, q_x, q_y):
    return tf.concat([p_x, p_y, tf.ones_like(p_x), tf.zeros_like(p_x), tf.zeros_like(p_x), tf.zeros_like(p_x), -p_x * q_x, -p_y * q_x], axis=1)


def ay(p_x, p_y, q_x, q_y):
    return tf.concat([tf.zeros_like(p_x), tf.zeros_like(p_x), tf.zeros_like(p_x), p_x, p_y, tf.ones_like(p_x), -p_x * q_y, -p_y * q_y], axis=1)


def homography(p1, p2, config):
    """
    This function is used to compute the homography between two sets of points p1 and p2. In our case. p1 is manually
    set to [[-1, -1], [1, -1], [1, 1], [-1, 1], p2 is set to the four corners of the quadrilateral region which is
    represented in the normalized image coordinate system (image center origin), i.e., [-1, 1] rather than [0, 1]

    p_2 = H \times p_1
    :param p1: [4, 2].  [[-1, -1], [1, -1], [1, 1], [-1, 1]
    :param p2: [Batch, 8]
    :return: [Batch, 9]
    """
    s = tf.shape(p2)
    p1 = tf.expand_dims(p1, 0)
    p1 = tf.tile(p1, [s[0], 1, 1])
    p1 = tf.reshape(p1, [s[0], -1])

    p2 = tf.reshape(p2, [s[0], -1])


    p1_lt_x, p1_lt_y, p1_rt_x, p1_rt_y, p1_rb_x, p1_rb_y, p1_lb_x, p1_lb_y = tf.split(p1, 8, axis=1)
    p2_lt_x, p2_lt_y, p2_rt_x, p2_rt_y, p2_rb_x, p2_rb_y, p2_lb_x, p2_lb_y = tf.split(p2, 8, axis=1)

    lt_x = ax(p1_lt_x, p1_lt_y, p2_lt_x, p2_lt_y)
    lt_y = ay(p1_lt_x, p1_lt_y, p2_lt_x, p2_lt_y)

    rt_x = ax(p1_rt_x, p1_rt_y, p2_rt_x, p2_rt_y)
    rt_y = ay(p1_rt_x, p1_rt_y, p2_rt_x, p2_rt_y)

    rb_x = ax(p1_rb_x, p1_rb_y, p2_rb_x, p2_rb_y)
    rb_y = ay(p1_rb_x, p1_rb_y, p2_rb_x, p2_rb_y)

    lb_x = ax(p1_lb_x, p1_lb_y, p2_lb_x, p2_lb_y)
    lb_y = ay(p1_lb_x, p1_lb_y, p2_lb_x, p2_lb_y)

    A = tf.stack([lt_x, lt_y, rt_x, rt_y, rb_x, rb_y, lb_x, lb_y], axis=1)
    # we build matrix A by using only 4 point correspondence. The linear
    # system is solved with the least square method, so here
    # we could even pass more correspondence

    P = tf.expand_dims(p2, axis=2)

    # here we solve the linear system
    # H = tf.matrix_solve(A, P)
    H = pseudo_matrix_solve(A, P, config)
    H = tf.squeeze(H, axis=-1)
    append_ones = tf.ones((s[0], 1), dtype=tf.float32)
    H_final = tf.concat([H, append_ones], axis=1)

    return H_final


############################################################
#  Pseudo linear algebra version of tensorflow functions
############################################################
def pseudo_matrix_solve(A, P, config):
    ATA = tf.matmul(A, A, transpose_a=True)
    ATP = tf.matmul(A, P, transpose_a=True)

    return tf.matmul(pseudo_inverse(ATA, config), ATP)

def pseudo_inverse(a, config, rcond=1e-15):
    a.set_shape((config.IMAGES_PER_GPU, 8, 8))
    s, u, v = tf.svd(a)
    # Ignore singular values close to zero to prevent numerical overflow
    limit = rcond * tf.reduce_max(s)
    non_zero = tf.greater(s, limit)

    reciprocal = tf.where(non_zero, tf.reciprocal(s), tf.zeros(K.shape(s)))
    lhs = tf.matmul(v, tf.matrix_diag(reciprocal))

    return tf.matmul(lhs, u, transpose_b=True)


def pseudo_determinant(a, rcond=1e-15):
    shape = K.int_shape(a)
    a.set_shape(shape)
    s, u, v = tf.svd(a)
    # Ignore singular values close to zero to prevent numerical overflow

    return tf.reduce_prod(s, axis=-1)


############################################################
#  Miscellaneous
############################################################
def convert_img_coord_to_grid_coord(img_coord, config):
    grid_coord = np.floor_divide(img_coord, config.GRID_SCALE)
    offsets = np.mod(img_coord, config.GRID_SCALE)
    offsets = (offsets - config.GRID_SCALE/2) / (config.GRID_SCALE/2)

    return grid_coord, offsets


def construct_3_3_matrix(a1, b1, c1, a2, b2, c2, a3, b3, c3):
    """
    Construct 9 elements in to a 3 by 3 matrix to compute the determinant
    The matrix is of  form |a1, b1, c1; a2, b2, c2; a3, b3, c3|
    :param a1: shape [batch, 1]
    :param b1: shape [batch, 1]
    :param c1: shape [batch, 1]
    :param a2: shape [batch, 1]
    :param b2: shape [batch, 1]
    :param c2: shape [batch, 1]
    :param a3: shape [batch, 1]
    :param b3: shape [batch, 1]
    :param c3: shape [batch, 1]
    :return:
    """
    first_row = tf.concat([a1, b1, c1], axis=1)
    second_row = tf.concat([a2, b2, c2], axis=1)
    third_row = tf.concat([a3, b3, c3], axis=1)

    matrix = tf.stack([first_row, second_row, third_row], axis=1)

    return matrix


def line_coefficient_graph(x_1, y_1, x_2, y_2):
    """
    Two points are given, how can we compute the line equation
    (x - x_1)/(x_2 - x_1) = (y - y_1)/(y_2 - y_1)
    (y_2 - y_1) * x - (y_2 - y_1) * x_1 = (x_2 - x_1) * y - (x_2 - x_1) * y_1
    (y_2 - y_1) * x + (x_1 - x_2) * y + (x_2 - x_1) * y_1 - (y_2 - y_1) * x_1
    So in the form of ax + by + c = 0
    a = (y_2 - y_1)
    b = (x_1 - x_2)
    c = (x_2 - x_1) * y_1 - (y_2 - y_1) * x_1
    :param x_1: The x cooridnate of p1 in tensor shape of [batch, 1]
    :param y_1: The y cooridnate of p1 in tensor shape of [batch, 1]
    :param x_2: The x cooridnate of p2 in tensor shape of [batch, 1]
    :param y_2: The y cooridnate of p2 in tensor shape of [batch, 1]
    :return: The coefficients of the line equation built by p1 and p2. ax + by + c = 0
    a: The first coefficient
    b  The y's coefficient:
    c: The c
    """
    a = (y_2 - y_1)
    b = (x_1 - x_2)
    c = (x_2 - x_1) * y_1 - (y_2 - y_1) * x_1
    return a, b, c


def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result


def norm_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


def denorm_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)

def denorm_cubes(cubes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (x, y)] in normalized coordinates
    shape: [..., (width, height)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = shape
    scale = np.array([w - 1, h - 1])
    return np.multiply(cubes, scale).astype(np.float32)


def recover_grid(grids):
    """
    transform predicted grids into a normalized image coordinates
    :param grids: The side or front_rear grid predicted by the model
    :return:
    """
    x = grids[0, :]
    y = grids[1, :]
    homo_append = grids[2, :]
    x_s = x/homo_append
    y_s = y/homo_append

    x_grid = (x_s + 1)*0.5
    y_grid = (y_s + 1)*0.5

    grids_coords = np.stack((x_grid, y_grid)).transpose()

    return grids_coords


def download_imagenet(backbone):
    """ Downloads ImageNet weights and returns path to weights file.
    """
    assert backbone in ['resnet50', 'resnet101'], 'The backbone name should be either resnet50 or resnet101'

    resnet_filename = 'ResNet-{}-model.keras.h5'
    resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)
    depth = int(backbone.replace('resnet', ''))

    filename = resnet_filename.format(depth)
    resource = resnet_resource.format(depth)
    if depth == 50:
        checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
    elif depth == 101:
        checksum = '05dc86924389e5b401a9ea0348a3213c'
    elif depth == 152:
        checksum = '6ee11ef2b135592f8031058820bb9e71'

    return get_file(
        filename,
        resource,
        cache_subdir='models',
        md5_hash=checksum
    )


def add_prefix_keras_model(prefix_name, keras_model):
    """
    Add name prefix to all the layers contained in a given keras model.
    For example, if we want to load two same ResNet50 into one module. If we do not modify one of them's name, the keras
    will throw a exception.

    :param prefix_name: Strings. the prefix name you want to add
    :param keras_model: a given keras model
    :return: renamed keras model.
    """
    layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers
    for layer in layers:
        # Is the layer a model?
        if layer.__class__.__name__ == 'Model':
            print("Adding prefix to sub keras model: ", layer.name)
            add_prefix_keras_model(prefix_name, keras_model=layer)
            continue

        original_layer_name = layer.name
        # Update layer. If layer is a container, update inner layer.
        if layer.__class__.__name__ == 'TimeDistributed':
            layer.layer.name = prefix_name + original_layer_name
        elif layer.__class__.__name__ == 'InputLayer':
            continue
        else:
            layer.name = prefix_name + original_layer_name

    return keras_model
