function p = multipleClassificationPredictionIndex(h)

    % Compute the prediction (the index of the max element).

    m = size(h, 1);
    p = zeros(m, 1);

    for i = 1:m
        [val, ind] = max(h(i, :));
        p(i) = ind;
    end

end