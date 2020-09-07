function p = multipleClassificationPredictionMatrix(h)

    % Compute the prediction (1 for the max element, 0 for the rest).

    m = size(h, 1);
    n = size(h, 2);
    p = zeros(m, n);

    for i = 1:m
        [val, ind] = max(h(i, :));
        p(i, ind) = 1;
    end

end