function theta = logisticRegressionGradientDescent(X, y, theta, alpha, num_iters, lambda)

    % Compute the modified theta taking num_iters steps' gradient descent with regularization.

    if nargin == 5
        lambda = 0;
    end

    m = length(y);

    z = X * theta;
    h = sigmoid(z);
    X0 = X(:, 1);
    X_without_X0 = X(:, 2:end);
    theta_without_theta0 = theta(2:end); % do not regularize theta(1)
    for iter = 1:num_iters
        grad0 = 1 / m * ((h - y)' * X0)'; 
        grad_rest = 1 / m * (((h - y)' * X_without_X0)' + lambda * theta_without_theta0);
        grad = [grad0; grad_rest];
        theta = theta - alpha * grad;
    end

end