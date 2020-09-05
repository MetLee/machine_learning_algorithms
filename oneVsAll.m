function all_theta = oneVsAll(X, y, num_labels)

    % Train num_labels logistic regression classifiers and return each of these classifiers in a matrix all_theta.

    m = size(X, 1);
    n = size(X, 2);
    all_theta = zeros(n, num_labels);

    for i = 1:num_labels
        binary_y = y == i;
        initial_theta = zeros(n, 1);
        options = optimset('GradObj', 'on', 'MaxIter', 50);
        theta = fmincg(@(t)(logisticRegressionCostF(X, binary_y, t)), initial_theta, options);
        all_theta(:, i) = theta;
    end

end