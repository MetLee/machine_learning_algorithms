function [error_train, error_cv] = validationCurve(X, y, X_cv, y_cv, lambda_vec, costFunction)

    % Generate the train and cross validation set errors needed to plot a validation curve for selecting lambda.

    n_lambda = size(lambda_vec);
    error_train = zeros(n_lambda, 1);
    error_cv = zeros(n_lambda, 1);

    for i = 1:length(lambda_vec)
        lambda = lambda_vec(i);
        theta = trainTheta(X, y, lambda, costFunction);
        error_train(i) = costFunction(X, y, theta, 0);
        error_cv(i) = costFunction(X_cv, y_cv, theta, 0);
    end

end