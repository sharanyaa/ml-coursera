function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


e_pow_neg_z = exp(-z);
% disp(e_pow_neg_z);

%using element wise division and addition
g = 1 ./ (1 .+ e_pow_neg_z);




% =============================================================

end
