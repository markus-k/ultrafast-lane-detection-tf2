import numpy as np


def generate_grid(lanes, cls_shape, image_shape, delete_lanes=None):    
    lanes = np.array(lanes)

    gt = np.zeros(cls_shape)

    for y in range(cls_shape[1]):
        yf = y / cls_shape[1]
        ly = int(lanes.shape[1] * yf)

        lx = lanes.T[ly,:]

        invalids = np.where(lx == -1)

        xf = lx / image_shape[1]
        x = np.round(xf * cls_shape[0]-1).astype(np.int)

        x[invalids] = cls_shape[0]-1

        if delete_lanes is not None:
            # delete unused lanes
            x = np.delete(x, (0,3))

        for i in range(len(x)):
            gt[x[i],y,i] = 1

    return gt
