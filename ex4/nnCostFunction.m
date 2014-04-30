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
X = [ones(m,1) X];         
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

% Completing part 1

% Initializing vars
  cost = 0;
% Computing the hypothesis value matrix. Feedforward algorithm.
  z_1 = X(:,2:end);
  a_1 = X;
  z_2 = a_1*Theta1';
  a_2 = sigmoid(z_2);
  a_2 = [ones(m,1) a_2];
  z_3 = a_2*Theta2';
  a_3 = sigmoid(z_3);
  h_theta = a_3;
 
% Looping through the m training examples.
for iter=1:m
% Creating the output vector for current training example.
  yCur = zeros(num_labels,1);
  for yIter = 1:num_labels
    if(yIter == y(iter))
      yCur(yIter) = 1;
    else
      yCur(yIter) = 0;
    end
  end
% Computing the cost for current training example, and adding it to net cost.
  % Computing the cost and incrementing value
   cost = cost + (-1/m)*sum( yCur.*log(h_theta(iter,:)') + (1.0-yCur).*log(1.0 - h_theta(iter,:)') );
end

%Adding cost due to regularization
tempTheta1 = Theta1;
tempTheta1(:,1) = 0;
tempTheta2 = Theta2;
tempTheta2(:,1) = 0;
cost = cost + (lambda/(2*m))*(sum(sum(tempTheta1.^2)) + sum(sum(tempTheta2.^2))); 
J = cost;

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

% We have already calculated the matrices a_1, a_2, a_3 in Feedforward part above. They include the column for the bias units.

% For each training example, we now implement Backpropagation algorithm. No delta_1 since no error in input units.
delta_3 = zeros(m,num_labels);
delta_2 = zeros(m,hidden_layer_size);

% Matrix for the BIG delta matrix corresponding to 
BIGDelta_1 = zeros(hidden_layer_size,input_layer_size+1);
BIGDelta_2 = zeros(num_labels,hidden_layer_size+1);
% Matrix for the BIG delta matrix corresponding to 
Theta1_grad = zeros(hidden_layer_size,input_layer_size+1);
Theta2_grad = zeros(num_labels,hidden_layer_size+1);

% Looping through training examples to find the error matrices for each training example.
for iter=1:m
  % Creating the output vector for current training example.
  yCur = zeros(num_labels,1);
  for yIter = 1:num_labels
    if(yIter == y(iter))
      yCur(yIter) = 1;
    else
      yCur(yIter) = 0;
    end
  end
  % Computing delta_3
  delta_3(iter,:) = a_3(iter,:) - yCur'; 
  
  % Computing delta_2 (Without bias units)
    % theta2 matrix ->*without*<- the column of params for bias units.
    theta2WB = Theta2(:,2:end);
    delta_2(iter,:) = theta2WB'*(delta_3(iter,:)').*sigmoidGradient(z_2(iter,:)');   
  
  % Accumulating the gradient terms.
   BIGDelta_1 = BIGDelta_1  + (delta_2(iter,:)')*(a_1(iter,:));
   BIGDelta_2 = BIGDelta_2  + (delta_3(iter,:)')*(a_2(iter,:));
 	  		
end

% Calculating the final gradient.
Theta1_grad = BIGDelta_1./m;
Theta2_grad = BIGDelta_2./m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
