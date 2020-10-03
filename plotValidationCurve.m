function [error_train, error_cv] = plotValidationCurve(X, y, X_cv, y_cv, costFunction, lambda_vec)

    % Plot the validation curve.

    [error_train, error_cv] = validationCurve(X, y, X_cv, y_cv, costFunction, lambda_vec);
    plot(lambda_vec, error_train, lambda_vec, error_cv);
    title('Validation curve');
    legend('Train', 'Cross Validation');
    xlabel('lambda');
    ylabel('Error');

end