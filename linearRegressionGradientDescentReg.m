function theta = linearRegressionGradientDescentReg(X, y, theta, alpha, lambda, num_iters)

    % Compute the modified theta taking num_iters steps' gradient descent with regularization.

    m = length(y);
    n_plus_1 = size(X, 2);
    
    theta_without_theta0 = theta(2:n_plus_1); % do not regularize theta(1)
    X_without_X0 = X(:, [2:n_plus_1]);
    for iter = 1:num_iters
        grad0 = 1 / m * ((X * theta - y)' * X)';
        grad_rest = 1 / m * (((h - y)' * X_without_X0)' + lambda * theta_without_theta0);
        grad = [grad0; grad_rest];
        theta = theta - alpha * grad;
    end

end