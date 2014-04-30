function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
  m = length(y); % number of training examples
  sumCost = 0;
% You need to return the following variables correctly 
% The implementation below uses vectorization.
for iter = 1:m
    xCur = [1; X(iter)];
    yCur = y(iter);
    sumCost = sumCost + ((theta'*xCur-yCur)^2);
% ====================== YOUR CODE HERE =========================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
% ===============================================================
end
J = sumCost/(2*m);
end
