function p = multipleClassificationPredictionIndex(h)

    % Compute the prediction (the index of the max element).

    [val, p] = max(h, [], 2);

end