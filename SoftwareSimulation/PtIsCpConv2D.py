## PT-IS-CP-dense/sparse dataflow proposed by SCNN implements.
import torch
from math import ceil, floor

def PT_IS_CP_Conv2D(feature:torch.tensor, weight:torch.tensor, stride=1, padding=True) -> torch.tensor:

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
    ## The output channels had a partition into K/Kc with quantity of Kc*R*S.
    ## The tiling scheme was applied to feature map in Ht and Wt with all input channels.
    ## Activation buffer and partial sum buffer had a capacity for handling an F*I metrix multiplication(MM).
    ## "F" and "I" represent to the MM size.
    F = 4 # Weight Vector
    I = 4 # Activation Vector
    Kc = 16
    Wt = 14
    Ht = 14
    
    ## We should be careful that the features or weights are contiguous in the row dimension. 
    WeightBuffer = torch.zeros(size=(C, ceil(Kc * R * S / F), F))
    FeatureBuffer = torch.zeros(size=(C, ceil(Ht * Wt / I), I))
    PartialSumBuffer = torch.zeros(size=(Kc, (Wt + R - 1), (Ht + S - 1)))
    ActivationBuffer = torch.zeros(size=(ceil(K/Kc), ceil(Kc*Wt*Ht)))

    for k_prime in range(ceil(K / Kc)):
        for c in range(C - 1):
            for a in range(ceil(Wt * Ht / I)):
                FeatureVector = FeatureBuffer[c, a, :].expand(F, I) # Fetch vector, broadcast
                for w in range(ceil(Kc * R * S / F)):
                    WeightVector = WeightBuffer[c, w, :].expand(F, I) # Fetch vector, broadcast
                    i = torch.tensor([i for i in range(I)]).expand(F, I) # Generate the indices, broadcast
                    f = torch.tensor([[i] for i in range(F)]).expand(F, I) # Generate the indices, broadcast

                    ## K, X, Y coordinates calculate, Store in "CooReg", Multiplier Array (F by I).
                    k = torch.floor((w * F + f) / (R * S))
                    x = torch.floor((a * I + i) / Wt) - torch.floor((w * F + f) % (R * S) / R) + floor(R / 2)
                    y = ((a * I + i) % Ht) - ((w * F + f) % S) + floor(S / 2)
                    #CooReg = torch.concat((k.unsqueeze(0), x.unsqueeze(0), y.unsqueeze(0)), dim=0).permute(1, 2, 0).contiguous().view(F*I, -1)

                    #print(f'Multiplier Array of K:\n{k}')
                    #print(f'Multiplier Array of X:\n{x}')
                    #print(f'Multiplier Array of Y:\n{y}')
                    #print(f'Multiplier Array of CooReg:\n{CooReg}')
                    
                    ## Values calculate, Store in "ValReg", Multiplier Array (F by I). 
                    v = FeatureVector * WeightVector
                    #print(f'v shape is {v.shape}')

                    ## Accumulating values to psum buffer follows the (k, x, y) through the all-to-all crossbar.
                    for acc_i in range(I):
                        for acc_f in range(F):
                            pk = int(k[acc_i, acc_f])
                            px = int(x[acc_i, acc_f])
                            py = int(y[acc_i, acc_f])
                            if(pk >= Kc or px < 0 or px >= (Wt + R - 1) or py < 0 or py >= (Ht + S - 1)): continue
                            else: PartialSumBuffer[pk, px, py] += v[acc_i, acc_f]


def FeatureTiling(feature:torch.tensor, PeParams:list) -> torch.tensor:
    ## Here, we define a single PE behavior with data storage.
    [H, W] = [feature.shape[2], feature.shape[3]]
    [Ht, Wt] = PeParams

    assert len(feature.shape) == 4, f'The requirement of convolutional feature was 4 dimension and \
        we found that only {len(feature.shape)} included.'
    
    ## Here, we slice the feature maps from large scale to several small tiles.
    ## The output shape of tiled feature maps is [TilingTags, InputChannels, Wt, Ht] (Transposed for PE processing).
    ## Total tiling tags are ceil(H / Ht) * ceil(W / Wt).
    TilingTags = ceil(H / Ht) * ceil(W / Wt)
    TiledFeatures = torch.zeros(size=(TilingTags, feature.shape[1], Wt, Ht))
    
    for i in range(int(TilingTags)):
        fh = int(floor(i / ceil(W / Wt)) * Ht)
        fw = int((i % ceil(W / Wt)) * Wt)

        if((fh + Ht) > feature.shape[2] or (fw + Wt) > feature.shape[3]):
            TiledFeatures[i, :, fw:feature.shape[3], fh:feature.shape[2]] = \
                feature[0, :, fh:feature.shape[2], fw:feature.shape[3]].transpose(2, 3)
        else:
            TiledFeatures[i] = feature[0, :, fh:(fh + Ht), fw:(fw + Wt)].transpose(2, 3)

    return TiledFeatures


def WeightSlicing(weight:torch.tensor, PeParams:list) -> torch.tensor:
    pass


## Functional Test
if __name__ == '__main__':

    feature = torch.rand(size=(1, 3, 64, 64))
    weight = torch.rand(size=(64, 3, 3, 3))










