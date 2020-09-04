function J = logisticRegressionCostReg(X, y, theta, lambda)

    % Compute the cost of logistic regression algorithm with regularization.

    m = length(y);
    n_plus_1 = size(X, 2);
    
    z = X * theta;
    h = sigmoid(z);
    costMatrix = - y .* log(h) - (1 - y) .* log(1 - h);
    theta_without_theta0 = theta(2:n_plus_1); % do not regularize theta(1)
    regMatrix = 1 / 2 * lambda * theta_without_theta0 .^ 2;
    J = 1 / m * (sum(costMatrix) + sum(regMatrix));

end