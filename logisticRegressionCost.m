function J = logisticRegressionCost(X, y, theta)

    % Compute the cost of logistic regression algorithm.

    m = length(y);
    
    z = X * theta;
    h = sigmoid(-z);
    costMatrix = - y .* log(h) - (1 - y) .* log(1 - h);
    J = 1 / m * sum(costMatrix);

end