function [J, grad] = costFunction(theta, X, y)

%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.
%   This function has the following assumptions.
%   1. X is provided without the first feature of ones.
%   2. theta is initialized to some value and has dimension that includes
%      the first feature of ones.

% Initialize some useful values
m = length(y); % number of training examples
nFeatures = length(theta);%Dimension of nFeatures.
% You need to return the following variables correctly 
J = 0;
grad = zeros(nFeatures,1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% For computing the cost function.

for iter=1:m
  hypothesisCalc = (theta')*(X(iter,:)');
  J = J + (-y(iter)*log(sigmoid(hypothesisCalc)) - (1-y(iter))*log(1-sigmoid(hypothesisCalc)))/m;
end


% For computing the gradient vector.
for iterOut =1:nFeatures
  for iter = 1:m
    hypothesisCalc = (theta')*(X(iter,:)');
    grad(iterOut) = grad(iterOut) + (sigmoid(hypothesisCalc) - y(iter))*X(iter,iterOut)/m; 
  end
end

% =============================================================

end
