from nnUtils import *

model = Sequential([
    SpatialConvolution(16,3,3,padding='SAME',name='bc_conv2d_1'),
    BatchNormalization('bn1'),
    ReLU(),
    Residual([
        SpatialConvolution(16,3,3,padding='SAME',name='bnn_conv2d_1'),
        BatchNormalization('bn2'),
        ReLU(),
        SpatialConvolution(16,3,3,padding='SAME',name='bnn_conv2d_2')
    ]),
    SpatialConvolution(32,3,3,padding='SAME',name='bnn_conv2d_3'),
    SpatialMaxPooling(2,2,2,2),
    BatchNormalization('bn3'),
    ReLU(),
    Residual([
        SpatialConvolution(32, 3, 3, padding='SAME', name='bnn_conv2d_4'),
        BatchNormalization('bn4'),
        ReLU(),
        SpatialConvolution(32, 3, 3, padding='SAME', name='bnn_conv2d_5')
    ]),
    SpatialConvolution(64,3,3,padding='SAME',name='bnn_conv2d_6'),
    SpatialMaxPooling(2,2,2,2),
    BatchNormalization('bn4'),
    ReLU(),
    Residual([
        SpatialConvolution(64, 3, 3, padding='SAME', name='bnn_conv2d_7'),
        BatchNormalization('bn5'),
        ReLU(),
        SpatialConvolution(64, 3, 3, padding='SAME', name='bnn_conv2d_8')
    ]),
    SpatialAveragePooling(8,8,1,1),
    BatchNormalization('bn6'),
    ReLU(),
    Affine(100, name='fc')
])
