function theta = logisticRegressionGradientDescent(X, y, theta, alpha, num_iters)

    % Compute the modified theta taking num_iters steps' gradient descent.

    m = length(y);

    z = X * theta;
    h = sigmoid(-z);
    for iter = 1:num_iters
        grad = 1 / m * ((h - y)' * X)';
        theta = theta - alpha * grad;
    end

end