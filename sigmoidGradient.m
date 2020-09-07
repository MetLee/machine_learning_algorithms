function grad = sigmoidGradient(z)

    % Compute the gradient of the sigmoid function.

    g = exp(-z) ./ (1 + exp(-z)) .^ 2;

end