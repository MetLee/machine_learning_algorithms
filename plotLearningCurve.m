function rst = plotLearningCurve(X, y, X_cv, y_cv, costFunction, lambda)

    % Plot the learning curve.

    if nargin == 5
        lambda = 0;
    end

    m = size(X, 1);

    [error_train, error_cv] = learningCurve(X, y, X_cv, y_cv, costFunction, lambda);
    plot(1:m, error_train, 1:m, error_cv);
    title('Learning curve for linear regression');
    legend('Train', 'Cross Validation');
    xlabel('Number of training examples');
    ylabel('Error');
    x_axis_max = m + 1;
    y_axis_max = max(max(error_train), max(error_cv)) + 10;
    axis([0 x_axis_max 0 y_axis_max]);

end