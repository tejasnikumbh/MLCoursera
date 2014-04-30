function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
for iter=1:size(z,1),
  for iter2 = 1:size(z,2),
    g(iter,iter2) =  1/(1 + exp(-z(iter,iter2)));
  end
end




% =============================================================

end
