function [J, grad] = linearRegressionCostF(X, y, theta)

    % Compute the cost and the gradient of linear regression algorithm.

    m = length(y);
    
    h = X * theta;
    costMatrix = h - y;
    J = 1 / (2 * m) * (costMatrix' * costMatrix);

    grad = 1 / m * ((h - y)' * X)';

end