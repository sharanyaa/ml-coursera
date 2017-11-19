function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%{ 
a1: 5000x401

z2: 5000x25

a2: 5000x26

a3: 5000x10

d3: 5000x10

d2: 5000x25

Theta1, Delta1 and Theta1_grad: 25x401

Theta2, Delta2 and Theta2_grad: 10x26

%}

%{
We'll start by outlining the forward propagation process. 
Though this was already accomplished once during Exercise 3, you'll need to 
duplicate some of that work because computing the gradients requires some of 
the intermediate results from forward propagation. 

Also, the y values in ex4 are a matrix, instead of a vector. 
This changes the method for computing the cost J.

1 - Expand the 'y' output values into a matrix of single values (see ex4.pdf Page 5). 
This is most easily done using an eye() matrix of size num_labels, 
with vectorized indexing by 'y'. A useful variable name would be "y_matrix"
%}

y_matrix = eye(num_labels)(y,:);

%{ 
Perform the forward propagation:

a1 equals the X input matrix with a column of 1's added (bias units) as the first column.
z2 equals the product of a1 and Θ1
a2 is the result of passing z2 through g()
Then add a column of bias units to a2 (as the first column).
NOTE: Be sure you DON'T add the bias units as a new row of Theta.
z3 equals the product of a2 and Θ2
a3 is the result of passing z3 through g()
%}

a1 = [ones(m, 1), X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1), a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
h_theta = a3;

%{ 
3 - Compute the unregularized cost according to ex4.pdf (top of Page 5), 
using a3, your y_matrix, and m (the number of training examples). 
Note that the 'h' argument inside the log() function is exactly a3. 
Cost should be a scalar value. 
Since y_matrix and a3 are both matrices, you need to compute the double-sum.

Remember to use element-wise multiplication with the log() function. 
For a discussion of why you can't (easily) use matrix multiplication here, see this thread:
https://www.coursera.org/learn/machine-learning/discussions/weeks/5/threads/ag_zHUGDEeaXnBKVQldqyw

Also, we're using the natural log, not log10().

%}

% first = -y_matrix' * log(a3);
% second = (1 .- y_matrix)' * log(1 .- a3);
% unregularized_cost = sum(sum((first - second)));
% J = (1/m) * unregularized_cost;
J = (-sum(sum(y_matrix.*log(h_theta))) - sum(sum((1-y_matrix).*(log(1-h_theta)))))/m;

%{ 
4 - Compute the regularized component of the cost according to ex4.pdf Page 6, 
using Θ1 and Θ2 (excluding the Theta columns for the bias units), 
along with λ, and m. The easiest method to do this is to compute 
the regularization terms separately, 
then add them to the unregularized cost from Step 3.
%}

% REGULARIZATION
regularization_term = (lambda/(2*m))*((sum(sum(Theta1(:,2:end).^2))) + sum(sum(Theta2(:,2:end).^2)));
J = J + regularization_term;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
%{ 
You can design your code for backpropagation based on analysis of 
the dimensions of all of the data objects. 
This tutorial uses the vectorized method, for easy comprehension and 
speed of execution.

Reference the four steps outlined on Page 9 of ex4.pdf.

---------------------------------

Let:

m = the number of training examples

n = the number of training features, including the initial bias unit.

h = the number of units in the hidden layer - NOT including the bias unit

r = the number of output classifications
%}

% Perform forward propagation -> done above

% d3 is the difference between a3 and the y_matrix. 
% The dimensions are the same as both, (m x r).
d3 = a3 - y_matrix;


% z2 came from the forward propagation process - it's the product of 
% a1 and Theta1, prior to applying the sigmoid() function. 
% Dimensions are (m x n) ⋅ (n x h) --> (m x h)

%  d2 is tricky. It uses the (:,2:end) columns of Theta2. 
% d2 is the product of d3 and Theta2(no bias), then 
% element-wise scaled by sigmoid gradient of z2. 
% The size is (m x r) ⋅ (r x h) --> (m x h). The size is the same as z2, as must be.
d2 = (d3 * Theta2(:, 2:end)) .* sigmoidGradient(z2);

% Delta1 is the product of d2 and a1. The size is (h x m) ⋅ (m x n) --> (h x n)
Delta1 = d2' * a1;

% Delta2 is the product of d3 and a2. 
% The size is (r x m) ⋅ (m x [h+1]) --> (r x [h+1])
Delta2 = d3' * a2;

% UNREGULARIZED GRADIENTS
% Theta1_grad and Theta2_grad are the same size as 
% their respective Deltas, just scaled by 1/m.
Theta1_grad = Theta1_grad + (1/m) * Delta1;
Theta2_grad = Theta2_grad + (1/m) * Delta2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%{ 
Since Theta1 and Theta2 are local copies, and we've already 
computed our hypothesis value during forward-propagation, 
we're free to modify them to make the gradient regularization easy to compute.
%}

% So, set the first column of Theta1 and Theta2 to all-zeros. 
Theta1(:, 1) = 0;
Theta2(:, 1) = 0;

% Scale each Theta matrix by λ/m and add each of these
% modified-and-scaled Theta matrices to the un-regularized Theta gradients 
% that were computed earlier.
Theta1_grad = Theta1_grad + (lambda/m) * Theta1;
Theta2_grad = Theta2_grad + (lambda/m) * Theta2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
