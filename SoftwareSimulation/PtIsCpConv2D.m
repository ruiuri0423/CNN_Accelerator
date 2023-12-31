% Function 2D with 7 dimension
function activation = PtIsCpConv2D(feature, weight)

    % Table for processing progress of convolution
    T = {'cycle', 'Output Channel', 'Output Column', 'Ouput Row'};
    cycle = 0;

    % Get the size of feature
    FeatureSize = size(feature);
    
    % Get the size of weight
    WeightSize = size(weight);
    
    % Checking the dimension of input array saturates for the convolutional
    % operation of CNN. [N, K, H, W] = [N, C, H, W] x [K, C, S, R].
    assert((length(FeatureSize) == 4), 'The dimension of feature is not suffienct for (N, C, H, W)');
    assert((length(WeightSize) == 4), 'The dimension of weight is not suffienct for (K, C, S, R)');
    assert((FeatureSize(2) == WeightSize(2)), 'The both feature and weight can not be perform convolution because of input channel(C) unequaled');
    
    % Creates the convolutional loop with 7 dimension
    N = FeatureSize(1); % Batches
    K = WeightSize(1);  % Output Channels
    C = FeatureSize(2); % Input Channels
    H = FeatureSize(3); % Feature Height
    W = FeatureSize(4); % Feature Width
    S = WeightSize(3);  % Weight Height
    R = WeightSize(4);  % Weight Width
    
    %% The PT-IS-CP-dense/sparse Dataflow:
    % Convolutional loop order: C -> W -> H -> K -> R -> S.
    % Output channels partialized for data reuse k_group = K/K_c.
    % For the parallelism, a tiling stategy is applied to the activation
    % plane with a W_t and H_t that scaled from W and H.
    
    F = 8; % Size of weight vector
    I = 8; % Size of feature vector
    K_c = 3; % Output channel partition
    W_t = 5; % Tile of width of feature
    H_t = 5; % tile of height of feature
    
    WeightBuffer = zeros([C, ceil(K_c*R*S/F), F]);
    FeatureBuffer = zeros([C, ceil(W_t*H_t/I), I]);
    AccumulationBuffer = zeros([K_c, (W_t+R-1), (H_t+S-1)]);
    OutputActivationBuffert = zeros([ceil(K/K_c), (K_c*W_t*H_t)]);
    
    for k_group = 0:ceil(K/K_c)-1
        for c = 0:C-1
            for a = 0:ceil(W_t*H_t/I)-1
                FeatureVector = FeatureBuffer(c+1, a+1, 1:I);
                for w = 0:ceil(K_c*R*S/F)-1
                    WeightVector = WeightBuffer(c+1, w+1, 1:F);
                    for i = 0:I-1
                        for f = 0:F-1
                            k = floor((w*F+f)/(R*S))+1; 
                            x = mod((a*I+i), W_t)-mod((w*F+f), R)+floor(R/2)+1;
                            y = floor((a*I+i)/W_t)-floor(mod((w*F+f), (R*S))/S)+floor(S/2)+1;
                            
                            if(x == 0 || x > W_t || y == 0 || y > H_t || k > K_c)
                                continue;
                            else
                                T((cycle+2), :) = {cycle, k-1, x-1, y-1};
                                cycle = cycle + 1;
                                AccumulationBuffer(k, x, y) = AccumulationBuffer(k, x, y) + FeatureVector(i+1) * WeightVector(f+1);
                            end
                        end
                    end
                end
            end
        end
        % OutputActivationBuffert(k_group+1, 1:(K_c*W_t*H_t-1)) = AccumulationBuffer(1:K_c, 1:W_t, 1:H_t);
    end
    
    writetable(table(T), 'ConvProc.xlsx');
end