function theta = linearRegressionGradientDescent(X, y, theta, alpha, num_iters)

    % Compute the modified theta taking num_iters steps' gradient descent.

    m = length(y);

    for iter = 1:num_iters
        grad = 1 / m * ((X * theta - y)' * X)';
        theta = theta - alpha * grad;
    end

end