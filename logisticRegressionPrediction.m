function p = logisticRegressionPrediction(X, theta)

    % Compute the prediction using a threshold at 0.5.

    m = size(X, 1);
    p = zeros(m, 1);
    
    z = X * theta;
    p = z >= 0; % sigmoid(z) >= 0.5 ==> Z >= 0

end