import numpy as np


def bounding_box(bounding_boxes):
    bboxes = np.dstack(bounding_boxes)
    bbox = np.zeros((2, 3), dtype=float)
    bbox[0, :] = bboxes[0, :, :].min(axis=-1)
    bbox[1, :] = bboxes[1, :, :].max(axis=-1)
    return bbox


class Geometry(object):
    def __init__(self):
        self.domain = None
        self.centroid = None

    def get_domain(self, force=False):
        """
        Returns coordinates of the smallest prism containing this geometry.

        :param force: if True, will force to (re)compute value (only necessary if geometry has changed)
        :type force: bool
        :return: opposite vertices of the bounding prism for this object
        :rtype: numpy.ndarray([[min], [max]])
        """
        raise NotImplementedError

    def obstructs(self, vector):
        """
        Returns true if the vector intersects with the current object

        TODO
        """
        raise NotImplementedError

    def get_centroid(self, force=False):
        """
        The centroid is considered the center point of the circumscribed
        parallelepiped, not the mass center.

        :param force: if True, will force to (re)compute value (only necessary if geometry has changed)
        :type force: bool
        :returns: (x, y, z) coordinates of the centroid of the object
        :rtype: numpy.ndarray
        """
        if force or self.centroid is None:
            self.centroid = self.get_domain().mean(axis=0)

        return self.centroid
