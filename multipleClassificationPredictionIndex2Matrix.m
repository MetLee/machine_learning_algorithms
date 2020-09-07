function p_m = multipleClassificationPredictionIndex2Matrix(p_i, num_labels)

    % Compute the prediction (1 for the tagged element, 0 for the rest).

    m = size(p_i, 1);
    p_m = zeros(m, num_labels);

    for i = 1:m
        p_m(i, p_i(i)) = 1;
    end

end