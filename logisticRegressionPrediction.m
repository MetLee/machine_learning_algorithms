function p = logisticRegressionPrediction(X, theta)

    % Compute the prediction using a threshold at 0.5.

    m = size(X, 1);
    p = zeros(m, 1);
    
    z = X * theta;
    for i = 1:m
        if z(i) >= 0 % sigmoid(z) >= 0.5 ==> Z >= 0
            p(i) = 1;
        else
            p(i) = 0;
        end
    end

end