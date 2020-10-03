function p = logisticRegressionPrediction(X, theta, threshold)

    % Compute the prediction.

    if nargin == 2
        threshold = 0.5;
    end

    m = size(X, 1);
    p = zeros(m, 1);
    
    z = X * theta;
    h = sigmoid(z);
    p = h >= threshold;

end