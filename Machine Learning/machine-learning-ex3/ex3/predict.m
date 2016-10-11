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


% The matrices Theta1 and Theta2 will now be in your Octave
% environment
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26
% X has size of 5000 x 400

% Add ones to the X data matrix
X = [ones(m, 1) X];

z2 = Theta1 * X';
A2 = sigmoid(z2);

% Add ones to the A2 data matrix
A2 = A2';
m2 = size(A2, 1);
A2 = [ones(m2, 1) A2];


z3 = Theta2 * A2';
h = sigmoid(z3);

[maxValues, indexes] = max(h', [], 2);
p = indexes;


% =========================================================================


end
