import tensorflow as tf


def sim_loss(y_true, y_pred):
    #assert y_pred.ndim == 4
    n, c, h, w = y_pred.shape
    losses = []
    
    for i in range(0, h-1):
        losses.append(y_pred[:,:,i,:] - y_pred[:,:,i+1,:])
    
    loss = tf.concat(losses, axis=0)
    
    # huber is smooth l1 loss
    huber = tf.keras.losses.Huber()
    return huber(loss, tf.zeros_like(loss))


def shape_loss(y_true, y_pred):
    n, c, h, w = y_pred.shape
    x = tf.math.softmax(y_pred[:,:c-1,:,:], axis=1)
    embedding = tf.reshape(tf.range(c-1, dtype=tf.float32), (1,-1,1,1))
    pos = tf.math.reduce_sum(x * embedding, axis=1)

    diff_list1 = []
    for i in range(0, h // 2):
        diff_list1.append(pos[:,i,:] - pos[:,i+1,:])
    
    # mae is not exaclty l1, but lets try
    l1 = tf.keras.losses.mae

    loss = 0
    for i in range(len(diff_list1)-1):
        loss += l1(diff_list1[i],diff_list1[i+1])
    loss /= len(diff_list1) - 1
    
    return loss


def ultrafast_loss(y_true, y_pred):
    a = 1.0
    b = 1.0
    
    cl = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    cl = tf.math.reduce_mean(cl) # not quite, but for now...
    
    sim = sim_loss(y_true, y_pred)
    shp = shape_loss(y_true, y_pred)
    return cl + a * sim + b * shp
