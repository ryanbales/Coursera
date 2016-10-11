function [J grad] = nnCostFunction(nn_params, ...
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

y_temp = zeros(m, num_labels);


for i = 1:m
    y_temp(i, y(i)) = 1;
endfor

% Add ones to the X matrix
X = [ones(size(X, 1), 1) X];

% Calculate tbe activations at Level 2
z2 = X * Theta1';
A2 = sigmoid(z2);

% Add ones to the A2 matrix
A2 = [ones(size(A2, 1), 1) A2];

% Calculate the activations at Level 3
z3 = A2 * Theta2';
H = sigmoid(z3);

% Calculate the Cost Function
S = sum(sum((y_temp .* log(H)) + ((1 - y_temp) .* log(1-H))));
J = (-1/m) * S;

% Regularize the Cost Function
R = sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2));
J = J + ((lambda/(2*m)) * R);



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

D2_gradient = zeros(size(Theta2));
D1_gradient = zeros(size(Theta1));

for t = 1:m
    % Back Prop - Part 1
    a1 = X(t, :);
    
    z2 = a1 * Theta1';
    a2 = [1 sigmoid(z2)];
    
    z3 = a2 * Theta2';
    a3 = sigmoid(z3);
    
   
    % Back Prop - Part 2
    y_t = y_temp(t, :);
    delta_3 =  a3 - y_t;
    
   
    % Back Prop - Part 3
    delta_temp = (delta_3 * Theta2);
    delta_2 = delta_temp .* [1 sigmoidGradient(z2)];
 
 
    % Back Prop - Part 4
    delta_2_gradient = delta_3' * a2;
    delta_1_gradient = delta_2(2:end)' * a1;
    
    D2_gradient = D2_gradient + delta_2_gradient;
    D1_gradient = D1_gradient + delta_1_gradient;
    
endfor

% Back Prop - Part 5
Theta1_grad = D1_gradient/m;
Theta2_grad = D2_gradient/m;






%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

R1 = (lambda/m) * Theta1(:, 2:end);
R2 = (lambda/m) * Theta2(:, 2:end);

Theta1_grad = Theta1_grad + [zeros(size(R1,1),1) R1];
Theta2_grad = Theta2_grad + [zeros(size(R2,1),1) R2];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
