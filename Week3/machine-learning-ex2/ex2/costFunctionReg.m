function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
z = X * theta; % m x 1 vector - size of y vector
htheta = sigmoid(z); % m x 1 vector

% First focus on the (y log) and (1-y log) portions of the cost equation. 
% Each of these is a vector of size (m x 1). 
% we'll distribute the summation operation, so we end up with two scalars
% The first term is the sum of -y multiplied by the natural log of h. 
% The natural log function is log(). Since we want the sum of 
% the products, we can use a vector multiplication. 
% The size of each argument is (m x 1), 
% we want the vector product to be a scalar, so use a transposition 
% so that (1 x m) times (m x 1) gives a result of (1 x 1), a scalar.

first = -y' * log(htheta);

% The blue-circled term uses the same method, 
% except that the two vectors are (1 - y) and the natural log of (1 - h).
second = (1 .- y)' * log(1 .- htheta);

% Subtract the right-side term from the left-side term
% Scale the result by 1/m. This is the unregularized cost.
unregularized_cost = 1/m * (first - second);

% Now we have only the regularization term remaining. 
% We want the regularization to exclude the bias feature, 
% so we can set theta(1) to zero. Since we already calculated h, 
% and theta is a local variable, we can modify theta(1) without 
% causing any problems.

theta(1) = 0;

% Now we need to calculate the sum of the squares of theta. 
% Since we've set theta(1) to zero, we can square the entire theta vector. 
% If we vector-multiply theta by itself, we will calculate the 
% sum automatically. So use the same method used before to 
% multiply theta by itself with a transposition.

sum_theta_square = theta' * theta;

% Now scale the cost regularization term by (lambda / (2 * m)). 
% Special Note for those whose cost value is too high: 1/(2*m) and (1/2*m) 
% give drastically different results.

regularized_cost_term = (lambda / (2*m)) * sum_theta_square;
J = unregularized_cost + regularized_cost_term

% Recall that the hypothesis vector htheta is the sigmoid() of 
% the product of X and ? (see ex2.pdf - Page 4). 
% Already calculated htheta for the cost J calculation.
% The left-side term is the vector product of X and (h - y), scaled by 1/m. 
% transpose and swap the product terms so the result is 
% (m x n)' times (m x 1) giving you a (n x 1) result. 
% This is the unregularized gradient. 
% Note that the vector product also includes the required summation.
unregularized_gradient = 1 / m * (X' * (htheta - y)); % n x 1 dimensions

% calculate the regularized gradient term as theta scaled by (lambda / m).
regularized_gradient = (lambda / m) * theta;

% The grad value is the sum of unregularized_gradient & regularized_gradient. 
% Since you forced theta(1) to be zero, the grad(1) term will only be 
% the unregularized value.
grad = unregularized_gradient + regularized_gradient

% =============================================================

end
