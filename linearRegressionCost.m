function J = linearRegressionCost(X, y, theta)

    % Compute the cost of linear regression algorithm.

    m = length(y);
    
    h = X * theta;
    costMatrix = h - y;
    J = 1 / (2 * m) * (costMatrix' * costMatrix);
    
end