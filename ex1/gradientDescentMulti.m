function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
diffCost = 0;
nFeatures = size(X,2) + 1;
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
    for iter1 = 1:m
      xCur = [1 X(iter1,:)]';
      % Note that differential cost is summed up for all training examples 
      % since the formula includes a summation.
      diffCost = alpha*((theta')*xCur - y(iter1))/m;
      theta= theta - xCur*diffCost;
    end
    % ============================================================
    %Note that theta vector is incremented once every iteration.
   

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end
end
