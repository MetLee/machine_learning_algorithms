function theta = linearRegressionNormalEquation(X, y)

    % Compute the theta using normal equation.

    theta = pinv(X' * X) * X' * y;

end