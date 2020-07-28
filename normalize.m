function X_norm = normalize(X)

    % Normalize X.
    % Use the standard deviation not the range.
    % X shouldn't contain the X_0 column.

    mu = mean(X, 1);
    sigma = std(X, 1);
    X_norm = (X - mu) ./ sigma;

end