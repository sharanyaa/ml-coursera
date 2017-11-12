function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Here is an outline for forward propagation using the vectorized method. 
% This is an implementation of the formula in Figure 2 on Page 11 of ex3.pdf.

% Add a column of 1's to X (the first column), and it becomes 'a1'.
% Multiply by Theta1 and you have 'z2'.
% Compute the sigmoid() of 'z2', then add a column of 1's, and 
% it becomes 'a2'
% Multiply by Theta2, compute the sigmoid() and it becomes 'a3'.
% Now use the max(a3, [], 2) function to return two vectors - 
% one of the highest value for each row, and one with its index. 
% Ignore the highest values. Keep the vector of the indexes where 
% the highest values were found. These are your predictions.
% Note: When you multiply by the Theta matrices, 
% you'll have to use transposition to get a result that is the correct size

% Note: The predictions must be returned as a column vector - size (m x 1). If you return a row vector, the script will not compute the accuracy correctly.

% Note: Not getting the correct results? 
% In the hidden layer, be sure you use sigmoid() first, 
% then add the bias unit.

% ------ dimensions of the variables ---------
% a1 is (m x n), where 'n' is the number of features including the bias unit
% Theta1 is (h x n) where 'h' is the number of hidden units
% a2 is (m x (h + 1))
% Theta2 is (c x (h + 1)), where 'c' is the number of labels.
% a3 is (m x c)
% p is a vector of size (m x 1)

% Theta1 and Theta2 - The parameters have dimensions
% that are sized for a neural network with 25 units in the second layer 
% and 10 output units (corresponding to the 10 digit classes).
% The matrices Theta1 and Theta2 will now be in your Octave environment

% Theta1 has size 25 x 401
% Theta2 has size 10 x 26
% Dimension of activation unit = #examples x #features(raw input or 
                                % hidden features) coming into that layer

a1 = [ ones(m, 1), X]; % a1 has dimensions m x (n+1) = 5000 x 401 ...
                                % m = # training examples (5000) ...
                                % n = # features (400)

% m
% [r_a1, c_a1] = size(a1)

z2 = a1 * Theta1'; % dim: m x h = 5000 x 25
% [r_z2, c_z2] = size(z2)

a2 = sigmoid(z2);
a2 = [ones(rows(a2), 1), a2];  % dim: m x (h+1) = 5000 x 26
% [r_a2, c_a2] = size(a2)

z3 = a2 * Theta2'; % z3 dim: 5000 x 10
% [r_z3, c_z3] = size(z3) 

a3 = sigmoid(z3); % dim: m x (num_labels) = 5000 x 10
% [r_a3, c_a3] = size(a3)

[max_hypotheses, p] = max(a3, [], 2);
# p = indices;
# [r, c] = size(p)


% =========================================================================


end
