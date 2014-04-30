function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
Carr = [0.1, 0.3 ,1,3,10,30,100];
sigmaarr = [0.01,0.03,0.1,0.3,1,3];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

count = 0;
for iter=1:length(Carr)
  for iter2=1:length(sigmaarr)
    model = svmTrain(X,y,Carr(iter),@(x1, x2) gaussianKernel(x1, x2, sigmaarr(iter2)));
    predictions = svmPredict(model,Xval);
    curError = mean(double(predictions ~= yval));
    if(count==0) 
      minError = curError;
      Copt = Carr(iter);
      sigmaopt = sigmaarr(iter2);
      count++;
    else
      if(curError<minError)
        minError = curError;
        Copt = Carr(iter);
        sigmaopt = sigmaarr(iter2);
      end
    end
  end
end

C = Copt;
sigma = sigmaopt;






% =========================================================================

end
