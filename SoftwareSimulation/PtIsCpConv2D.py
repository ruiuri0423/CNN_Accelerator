## PT-IS-CP-dense/sparse dataflow proposed by SCNN implements.
import torch

def PT_IS_CP_Conv2D(feature:torch.tensor, weight:torch.tensor, stride=1, padding=True):

    ## Here, we define a single PE behavior with data storage.
    FeatureShape = feature.shape
    WeightShape = weight.shape

    assert len(FeatureShape) == 4, f'The requirement of convolutional feature was 4 dimension and \
        we found that only {len(FeatureShape)} included.'
    assert len(WeightShape) == 4, f'The requirement of convolutional weight was 4 dimension and \
        we found that only {len(WeightShape)} included.'
    assert FeatureShape[1] == WeightShape[1], f'The input channel between feature and weight should \
        have same quntities.'

    ## The convolution parameters were defined by SCNN. 
    N = FeatureShape[0] # Batches
    K = WeightShape[0] # Output Channels
    C = FeatureShape[1] # Input Channels
    H = FeatureShape[2] # Feature Height
    W = FeatureShape[3] # Feature Width
    S = WeightShape[2] # Kernel Height
    R = WeightShape[3] # Kernel Width

    ## Basic loop topology of PT-IS-CP-dense was N > C > W > H > K > R > S.
    ## The Height (rows) was processed first in both feature and weight.
    ## The output channels had a partition into K/K_c with quantity of K_c*R*S.
    ## The tiling scheme was applied to feature map in H_t and W_t with all input channels.
    ## Activation buffer and partial sum buffer had a capacity for handling an F*I metrix multiplication(MM).
    ## "F" and "I" represent to the MM size.
    F = 4 # Weight Vector
    I = 4 # Activation Vector
    K_c = 16
    W_t = 14
    H_t = 14
    
    WeightBuffer = torch.zeros(shape=())
    FeatureBuffer = torch.zeros(shape=())
    PatialSumBuffer = torch.zeros(shape=())
    ActivationBuffer = torch.zeros(shape=())






