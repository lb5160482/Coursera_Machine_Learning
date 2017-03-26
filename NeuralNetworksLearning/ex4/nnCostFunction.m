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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% 1. cost computation
X = [ones(size(X, 1),1), X]; % add x0 feature
a1 = X';
z2 = Theta1 * a1;
a2 = [ones(1, size(z2, 2)); sigmoid(z2)];
z3 = Theta2 * a2;
hx = (sigmoid(z3))'; % m x num_labels
yVec = zeros(m, num_labels);
for i = 1 : m
	yVec(i, y(i)) = 1;
end
JMat = -yVec .* log(hx) - (1 - yVec) .* log(1 - hx);
J = sum(sum(JMat)) / m;


%% use for loop
% for i = 1 : m
% 	yi = zeros(num_labels, 1);
% 	yi(y(i)) = 1;
% 	a1 = X(i, :)';
% 	z2 = Theta1 * a1;
% 	a2 = [1;sigmoid(z2)];
% 	z3 = Theta2 * a2;
% 	hx = sigmoid(z3);
% 	J = J + (- yi' * log(hx) - (1 - yi)' * log(1 - hx));
% end
% J = J / m;

% 2. Regularized cost function
J = J + (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2))) * lambda / 2 / m; % should not include theta0 for x0!

% 3. Backpropagation
for i = 1 : m
	% forward propagation to compute output layer
	a1 = X(i, :)';
	z2 = Theta1 * a1;
	a2 = [1; sigmoid(z2)];
	z3 = Theta2 * a2;
	output = sigmoid(z3);
	% compute deltaL
	yL = ([1 : num_labels] == y(i))';
	delta3 = output - yL;
	% compute delta2
	delta2 = Theta2' * delta3 .* [1;sigmoidGradient(z2)];
	delta2 = delta2(2 : end);
	% compute D
	Theta1_grad = Theta1_grad + delta2 * a1';
	Theta2_grad = Theta2_grad + delta3 * a2';
end
% compute gradient for each theta
Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% regularization
regTerm1 = lambda / m * Theta1(:, 2 : end);
regTerm2 = lambda / m * Theta2(:, 2 : end);
Theta1_grad(:, 2 : end) = Theta1_grad(:, 2 : end) + regTerm1;
Theta2_grad(:, 2 : end) = Theta2_grad(:, 2 : end) + regTerm2;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
