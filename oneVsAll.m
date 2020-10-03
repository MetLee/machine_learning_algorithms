function all_theta = oneVsAllReg(X, y, num_labels, lambda)

    % Train num_labels logistic regression classifiers with regularization and return each of these classifiers in a matrix all_theta.

    if nargin == 3
        lambda = 0;
    end

    m = size(X, 1);
    n = size(X, 2);
    all_theta = zeros(n, num_labels);

    for i = 1:num_labels
        binary_y = y == i;
        initial_theta = zeros(n, 1);
        options = optimset('GradObj', 'on', 'MaxIter', 50);
        theta = fmincg(@(t)(logisticRegressionCost(X, binary_y, t, lambda)), initial_theta, options);
        all_theta(:, i) = theta;
    end

end