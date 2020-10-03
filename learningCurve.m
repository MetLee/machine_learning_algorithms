function [error_train, error_cv] = learningCurve(X, y, X_cv, y_cv, costFunction, lambda)

    % Generate the train and cross validation set errors needed to plot a learning curve.

    if nargin == 5
        lambda = 0;
    end

    m = size(X, 1);
    error_train = zeros(m, 1);
    error_cv = zeros(m, 1);

    for i = 1:m
        theta = trainTheta(X(1:i, :), y(1:i, :), lambda, costFunction);
        [error_train(i)] = costFunction(X(1:i, :), y(1:i, :), theta, 0);
        [error_cv(i)] = costFunction(X_cv, y_cv, theta, 0);
    end

end