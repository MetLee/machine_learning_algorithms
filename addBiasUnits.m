function matrix_out = addBiasUnits(matrix_in)

    % Add bias units.

    m = size(matrix_in, 1);

    matrix_out = [ones(m, 1) matrix_in];

end