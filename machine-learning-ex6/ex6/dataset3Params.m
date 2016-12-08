function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 1;

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
params = [0.01 0.03 0.1 0.3 1 3 10 30];
nParams = length(params);

Ctest = 0;
sigmatest = 0;
predictions_errors = zeros(nParams,nParams);
for k=1:nParams
    Ctest = params(k);
    for j=1:nParams
        sigmatest = params(j);
        model = svmTrain(X, y, Ctest, @(x1, x2) gaussianKernel(x1, x2, sigmatest));
        predictions = svmPredict(model,Xval);
        predictions_errors(k,j) = mean(double(predictions ~= yval));
    end
end
[Cidx, sigmaidx] = find(predictions_errors==min(min(predictions_errors)));
C = params(Cidx);
sigma =params(sigmaidx);



% =========================================================================

end
