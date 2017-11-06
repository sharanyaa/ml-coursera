function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta

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

first = -y' * log(htheta)

% The blue-circled term uses the same method, 
% except that the two vectors are (1 - y) and the natural log of (1 - h).
second = (1 .- y)' * log(1 .- htheta)

% Subtract the right-side term from the left-side term
% Scale the result by 1/m. This is the unregularized cost.
J = 1/m * (first - second)

% Recall that the hypothesis vector htheta is the sigmoid() of 
% the product of X and ? (see ex2.pdf - Page 4). 
% Already calculated htheta for the cost J calculation.
% The left-side term is the vector product of X and (h - y), scaled by 1/m. 
% transpose and swap the product terms so the result is 
% (m x n)' times (m x 1) giving you a (n x 1) result. 
% This is the unregularized gradient. 
% Note that the vector product also includes the required summation.
grad = 1 / m * (X' * (htheta - y)) % n x 1 dimensions

% =============================================================

end
