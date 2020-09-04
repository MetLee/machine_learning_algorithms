function [J, grad] = logisticRegressionCostF(X, y, theta)

    % Compute the cost and the gradient of logistic regression algorithm.

    m = length(y);
    
    z = X * theta;
    h = sigmoid(z);
    costMatrix = - y .* log(h) - (1 - y) .* log(1 - h);
    J = 1 / m * sum(costMatrix);

    grad = 1 / m * ((h - y)' * X)';

end