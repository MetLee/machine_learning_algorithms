function p = oneVsAllPrediction(X, theta)

    % Compute the prediction (the index of the max element).

    m = size(X, 1);
    num_labels = size(theta, 2);
    p = zeros(m, 1);

    z = X * theta;
    for i = 1:m
        [val, ind] = max(z(i, :));
        p(i) = ind;
    end

end