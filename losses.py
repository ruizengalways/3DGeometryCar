import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
from fgcn import utils
import numpy as np

############################################################
#  Loss Functions
############################################################
def focal(alpha=0.25, gamma=2.0):
    """ Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """
    def _focal(y_true, y_pred):
        """ Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        labels         = y_true
        classification = y_pred

        # filter out "ignore" anchors
        anchor_state   = keras.backend.max(labels, axis=2)  # -1 for ignore, 0 for background, 1 for object
        indices        = keras.backend.where(keras.backend.not_equal(anchor_state, -1))
        labels         = keras.backend.gather_nd(labels, indices)
        classification = keras.backend.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = keras.backend.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = keras.backend.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(1.0, normalizer)

        return keras.backend.sum(cls_loss) / normalizer

    return _focal



def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typicallly: [N, 4], but could be any shape.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def subnet_cls_loss_graph(y_true, y_pred, alpha=0.25, gamma=2.0):
    """ Compute the focal loss given the target tensor and the predicted tensor.

     As defined in https://arxiv.org/abs/1708.02002

     Args
         y_true: Tensor of target data from the generator with shape (B, N, num_classes).
         y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

     Returns
         The focal loss of y_pred w.r.t. y_true.
     """
    labels = y_true
    classification = y_pred

    # filter out "ignore" anchors
    anchor_state = K.max(labels, axis=2)  # -1 for ignore, 0 for background, 1 for object
    indices = tf.where(K.not_equal(anchor_state, -1))
    labels = tf.gather_nd(labels, indices)
    classification = tf.gather_nd(classification, indices)

    # compute the focal loss
    alpha_factor = K.ones_like(labels) * alpha
    alpha_factor = tf.where(K.equal(labels, 1), alpha_factor, 1 - alpha_factor)
    focal_weight = tf.where(K.equal(labels, 1), 1 - classification, classification)
    focal_weight = alpha_factor * focal_weight ** gamma

    cls_loss = focal_weight * K.binary_crossentropy(labels, classification)

    # compute the normalizer: the number of positive anchors
    normalizer = tf.where(K.equal(anchor_state, 1))
    normalizer = K.cast(K.shape(normalizer)[0], tf.float32)
    normalizer = K.maximum(1.0, normalizer)

    return K.sum(cls_loss) / normalizer


def subnet_reg_loss_graph(y_true, y_pred, input_anchor_state, sigma=3.0):
    """Return the Subnet 2D bounding box loss graph.

    y_true: target_bbox, [batch, the number of all anchors, (dy, dx, log(dh), log(dw))].
    y_pred: predicted 2D bbox, [batch, the number of all anchors, (dy, dx, log(dh), log(dw)]..
    input_anchor_state: the state of each anchor, The value is in
        we only use positive [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    sigma_squared = sigma ** 2
    regression = y_pred
    regression_target = y_true
    anchor_state = input_anchor_state

    # filter out "ignore" anchors
    indices = tf.where(K.equal(anchor_state, 1))
    regression = tf.gather_nd(regression, indices)
    regression_target = tf.gather_nd(regression_target, indices)

    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
    regression_diff = regression - regression_target
    regression_diff = K.abs(regression_diff)
    regression_loss = tf.where(
        K.less(regression_diff, 1.0 / sigma_squared),
        0.5 * sigma_squared * K.pow(regression_diff, 2),
        regression_diff - 0.5 / sigma_squared
    )

    # compute the normalizer: the number of positive anchors
    normalizer = K.maximum(1, K.shape(indices)[0])
    normalizer = K.cast(normalizer, dtype=tf.float32)
    return K.sum(regression_loss) / normalizer


def cube_reg_loss_graph(y_true, y_pred, detected_boxes, sigma=3.0):
    """Return the cube corners regression loss graph.

    y_true: target_cube_reg, [batch, 8, (x, y)]. Note: the cube corners coordinate is in 2d bounding box coordinate
        system
    y_pred: predicted cube regression, [batch, 8, (x, y)].

    detected_boxes: This variable is used to create a index to filter out illegitimate 2d bounding box. In some extreme
        case, the 2D bounding box cannot be detected via RetinaNetwork. So the output of this sort of images could be
        [-1, -1, -1, -1], We need to locate it and neutralize its contribution to the cube loss.

    Returns:
        loss: cube_reg_loss. cube regression smooth l1 loss.
    """
    # Filter out the image we cannot detect any 2D bounding box, i.e., we only use the images where 2D bounding box can
    #   be detected to train cube_reg_loss and cube_vp_loss

    detected_boxes = tf.squeeze(detected_boxes, axis=1)
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(detected_boxes), axis=1), tf.bool)

    y_true = tf.boolean_mask(y_true, non_zeros)
    y_pred = tf.boolean_mask(y_pred, non_zeros)

    sigma_squared = sigma ** 2
    # Mathematic:
    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0/sigma_squared), "float32")
    loss = (less_than_one * 0.5 * sigma_squared * diff**2) + (1 - less_than_one) * (diff - 0.5 / sigma_squared)
    loss = K.mean(loss)

    return loss


def cube_vp_loss_graph(pred_cube_coords, detected_boxes, config):
    """Loss for the 3D bounding box regression for cube model.
    pred_cube_coords: [batch, num_x_y_offsets (2*8)], The offsets of eight corners to the corresponding grid center.
        Because eight corners of 3D bounding box have been transformed in 2D bounding box coordinate system.
         So the grid center is the center of the 2D bounding box.

    detected_boxes: This variable is used to create a index to filter out illegitimate 2d bounding box. In some extreme
    case, the 2D bounding box cannot be detected via RetinaNetwork. So the output of this sort of images could be
    [-1, -1, -1, -1], We need to locate it and neutralize its contribution to the cube loss.
    """
    # Filter out the image we cannot detect any 2D bounding box, i.e., we only use the images where 2D bounding box can
    #   be detected to train cube_reg_loss and cube_vp_loss
    detected_boxes = tf.squeeze(detected_boxes, axis=1)
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(detected_boxes), axis=1), tf.bool)
    pred_cube_coords = tf.boolean_mask(pred_cube_coords, non_zeros)

    p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, p5_x, p5_y, p6_x, p6_y, p7_x, p7_y =\
        tf.split(pred_cube_coords, 16, axis=1)

    # The lines generated from front to rear to its opposite
    line_p0p3_a, line_p0p3_b, line_p0p3_c = utils.line_coefficient_graph(p0_x, p0_y, p3_x, p3_y)
    line_p1p2_a, line_p1p2_b, line_p1p2_c = utils.line_coefficient_graph(p1_x, p1_y, p2_x, p2_y)
    line_p4p7_a, line_p4p7_b, line_p4p7_c = utils.line_coefficient_graph(p4_x, p4_y, p7_x, p7_y)
    line_p5p6_a, line_p5p6_b, line_p5p6_c = utils.line_coefficient_graph(p5_x, p5_y, p6_x, p6_y)

    matrix_front_rear1 = utils.construct_3_3_matrix(line_p0p3_a, line_p0p3_b, line_p0p3_c,
                                                    line_p1p2_a, line_p1p2_b, line_p1p2_c,
                                                    line_p4p7_a, line_p4p7_b, line_p4p7_c)
    det_matrix_front_rear1 = utils.pseudo_determinant(matrix_front_rear1)

    matrix_front_rear2 = utils.construct_3_3_matrix(line_p5p6_a, line_p5p6_b, line_p5p6_c,
                                                    line_p1p2_a, line_p1p2_b, line_p1p2_c,
                                                    line_p4p7_a, line_p4p7_b, line_p4p7_c)
    det_matrix_front_rear2 = utils.pseudo_determinant(matrix_front_rear2)

    # The lines generated from side to its opposite
    line_p0p1_a, line_p0p1_b, line_p0p1_c = utils.line_coefficient_graph(p0_x, p0_y, p1_x, p1_y)
    line_p4p5_a, line_p4p5_b, line_p4p5_c = utils.line_coefficient_graph(p4_x, p4_y, p5_x, p5_y)
    line_p3p2_a, line_p3p2_b, line_p3p2_c = utils.line_coefficient_graph(p3_x, p3_y, p2_x, p2_y)
    line_p7p6_a, line_p7p6_b, line_p7p6_c = utils.line_coefficient_graph(p7_x, p7_y, p6_x, p6_y)

    matrix_side1 = utils.construct_3_3_matrix(line_p0p1_a, line_p0p1_b, line_p0p1_c,
                                              line_p4p5_a, line_p4p5_b, line_p4p5_c,
                                              line_p7p6_a, line_p7p6_b, line_p7p6_c)

    det_matrix_side1 = utils.pseudo_determinant(matrix_side1)

    matrix_side2 = utils.construct_3_3_matrix(line_p0p1_a, line_p0p1_b, line_p0p1_c,
                                              line_p3p2_a, line_p3p2_b, line_p3p2_c,
                                              line_p7p6_a, line_p7p6_b, line_p7p6_c)

    det_matrix_side2 = utils.pseudo_determinant(matrix_side2)


    # The lines generated from roof to its opposite
    line_p0p4_a, line_p0p4_b, line_p0p4_c = utils.line_coefficient_graph(p0_x, p0_y, p4_x, p4_y)
    line_p1p5_a, line_p1p5_b, line_p1p5_c = utils.line_coefficient_graph(p1_x, p1_y, p5_x, p5_y)
    line_p3p7_a, line_p3p7_b, line_p3p7_c = utils.line_coefficient_graph(p3_x, p3_y, p7_x, p7_y)
    line_p2p6_a, line_p2p6_b, line_p2p6_c = utils.line_coefficient_graph(p2_x, p2_y, p6_x, p6_y)

    matrix_roof1 = utils.construct_3_3_matrix(line_p0p4_a, line_p0p4_b, line_p0p4_c,
                                              line_p1p5_a, line_p1p5_b, line_p1p5_c,
                                              line_p3p7_a, line_p3p7_b, line_p3p7_c)
    det_matrix_roof1 = utils.pseudo_determinant(matrix_roof1)

    matrix_roof2 = utils.construct_3_3_matrix(line_p2p6_a, line_p2p6_b, line_p2p6_c,
                                              line_p1p5_a, line_p1p5_b, line_p1p5_c,
                                              line_p3p7_a, line_p3p7_b, line_p3p7_c)
    det_matrix_roof2 = utils.pseudo_determinant(matrix_roof2)

    loss = (tf.abs(det_matrix_front_rear1) + tf.abs(det_matrix_front_rear2) + \
           tf.abs(det_matrix_side1) + tf.abs(det_matrix_side2) + \
           tf.abs(det_matrix_roof1) + tf.abs(det_matrix_roof2))/6

    loss = K.mean(loss)

    return loss


def final_class_loss_graph(target_class_ids, pred_class_logits, detected_boxes):
    """Loss for the last classifier layer of FGCN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    outlier_index: Used to extract inliers
    """
    # During model building, Keras calls this function with
    # target_class_ids of type float32. Unclear why. Cast it
    # to int to get around it.
    target_class_ids = tf.squeeze(target_class_ids, axis=1)
    detected_boxes = tf.squeeze(detected_boxes, axis=1)
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(detected_boxes), axis=1), tf.bool)

    target_class_ids = tf.boolean_mask(target_class_ids, non_zeros)
    target_class_ids = tf.cast(target_class_ids, 'int64')


    numbers_inliers = tf.cast(tf.reduce_sum(tf.cast(non_zeros, tf.int64), axis=0), tf.int32)
    pred_class_logits = pred_class_logits[0:numbers_inliers, :]

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    loss = K.mean(loss)

    return loss
