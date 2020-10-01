function [theta] = trainTheta(X, y, lambda, costFunction)

    % Train theta.

    initial_theta = zeros(size(X, 2), 1);
    options = optimset('MaxIter', 200, 'GradObj', 'on');
    theta = fmincg(@(t) costFunction(X, y, t, lambda), initial_theta, options);

end