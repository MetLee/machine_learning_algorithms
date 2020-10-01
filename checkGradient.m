function numGrad = computeNumericalGradient(JFunction, theta)
    
    % Computes the gradient using "finite differences".

    numGrad = zeros(size(theta));
    perturb = zeros(size(theta));
    epsilon = 1e-4;

    for p = 1:numel(theta)
        % Set perturbation vector
        perturb(p) = epsilon;
        costMinus = JFunction(theta - perturb);
        costPlus = JFunction(theta + perturb);
        % Compute Numerical Gradient
        numGrad(p) = (costPlus - costMinus) / (2 * epsilon);
        perturb(p) = 0;
    end

end


function rst = checkGradient(JFunction, theta)

    % Check the gradients.

    [cost, grad] = JFunction(theta);
    numGrad = computeNumericalGradient(JFunction, theta);
    diff = norm(numGrad-grad) / norm(numGrad+grad);

    if diff < 1e-9
        return true;
    else
        return false;
    end

end