function theta = initializeTheta(m, n, epsilon)

    % Initialize theta randomly.

    theta = rand(m, n) * (2 * epsilon) - epsilon;

end