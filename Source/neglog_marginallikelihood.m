function [result] = neglog_marginallikelihood(theta)
global hyp_xtrain
global y0_hyp_ytrain_pog
% global noise_prior
% sigma=noise_prior;
%sigma = sqrt(0.005);
%taken from spgp_pred.m from pI code
%     X=hyp_xtrain';
%     Y=hyp_xtrain';
    X=hyp_xtrain;
    Y=hyp_xtrain;
    [n1,dim] = size(X); n2 = size(Y,1);
%      b = exp(theta(1:end-1)); c = exp(theta(end));
%   b = exp(theta(1:end-2)); c = exp(theta(end-1)); 
  b = theta(1:end-2); c = theta(end-1); 
   sigma=theta(end);
    X = X.*repmat(sqrt(b)',n1,1);
    Y = Y.*repmat(sqrt(b)',n2,1);
    
    K = -2*X*Y' + repmat(sum(Y.*Y,2)',n1,1) + repmat(sum(X.*X,2),1,n2);
    K = c*exp(-0.5*K);
%     sigmasquare_identity=K+(sigma^2)*eye(n1);
    sigmasquare_identity=K+(sigma)*eye(n1);
%     result=-log(exp(-0.5*y0_hyp_ytrain_pog*(sigmasquare_identity)^(-1)*y0_hyp_ytrain_pog')/sqrt((2*pi)^(n1)*det(sigmasquare_identity)));
%     result=-log(exp(-0.5*y0_hyp_ytrain_pog'*(sigmasquare_identity)^(-1)*y0_hyp_ytrain_pog)); %/sqrt((2*pi)^(n1)*det(sigmasquare_identity)));
    result = 0.5*y0_hyp_ytrain_pog'*(sigmasquare_identity)^(-1)*y0_hyp_ytrain_pog + 0.5*log(det(sigmasquare_identity)) + 0.5*log(2*pi)*n1;
    
end