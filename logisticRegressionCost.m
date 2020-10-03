function [J, grad] = logisticRegressionCost(X, y, theta, lambda)

    % Compute the cost and the gradient of logistic regression algorithm with regularization.

    if nargin == 3
        lambda = 0;
    end

    m = length(y);
    
    z = X * theta;
    h = sigmoid(z);
    costMatrix = - y .* log(h) - (1 - y) .* log(1 - h);
    theta_without_theta0 = theta(2:end); % do not regularize theta(1)
    regMatrix = lambda * theta_without_theta0 .^ 2;
    J = 1 / m * (sum(costMatrix) + 1 / 2 * sum(regMatrix));

    X0 = X(:, 1);
    X_without_X0 = X(:, 2:end);
    grad0 = 1 / m * ((h - y)' * X0)';
    grad_rest = 1 / m * (((h - y)' * X_without_X0)' + lambda * theta_without_theta0);
    grad = [grad0; grad_rest];

end