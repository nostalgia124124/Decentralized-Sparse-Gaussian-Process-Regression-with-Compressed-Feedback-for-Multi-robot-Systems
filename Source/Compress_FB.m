%Compress_FB.m
%
%DESCRIPTION:
%    implements data compress with the feedbacks of decentralized Gaussian
%    posterior distributions
%
%INPUTS:
%    *D & y: compress dictionary 
%    *x: text datasets
%    *kernel: Gaussian kernel 
%    *eps2: squared-norm distance for stopping criterion
%    *fused_mu & fused_sigma: decentralized Gaussian posterior
%    distributions
%
%OUTPUTS:
%    *idxdkmppf: indices of dictionary points used (refer to columns of D)


function idxdkmppf = Compress_FB(D,y,kernel,x,eps2,noise_prior,fused_mu,fused_sigma)

% the set of indices of D that we are keeping
Y = 1:size(y,2);
N = length(Y);
% the set of indices of D that we are removing
Z = [];
ytmp = y;

KDD=kernel.f(D,D);
k_XX=kernel.f(x,x);
k_DX=kernel.f(D,x);

 %compute candidate posterior distribution parameters
    mu_full=k_DX'/(KDD + noise_prior^2*eye(size(KDD)))*y';
    Sigma_full=diag(k_XX-k_DX'/(KDD + noise_prior^2*eye(size(KDD)))*k_DX + noise_prior^2);

    HD_FD = hellinger_distance(fused_mu,mu_full,diag(fused_sigma),diag(Sigma_full));

% continue removing points greedily untill Hellinger hits metrics
continue_pruning = 1; 
while continue_pruning
  gmin = Inf;

  for i = 1:length(Y)
      % removal set to consider
      Zi = [Z Y(i)];
      remains=setdiff(1:N,Zi);

      % compute mean and covariance with index Zi removed 
      mu_removal=k_DX(remains,:)'/(KDD(remains,remains) ...
                + noise_prior^2*eye(size(KDD(remains,remains))))*ytmp(remains)';
      Sigma_removal=diag(k_XX-k_DX(remains,:)'/(KDD(remains,remains) ...
                + noise_prior^2*eye(size(KDD(remains,remains))))*k_DX(remains,:)+ noise_prior^2);
      
      % compute error for this removal set, see if smallest
      HD_Fdj = hellinger_distance(fused_mu,mu_removal,diag(fused_sigma),diag(Sigma_removal));
      gi = abs(HD_Fdj - HD_FD);

      if gi < gmin
          gmin = gi;
          imin = i;
      end

  end
  % if best error is still okay, delete the corresponding element
  if gmin <= eps2
    Z = [Z Y(imin)];
    Y(imin) = [];

  % otherwise, stop
  else
    continue_pruning = 0;
    % return the indices that we kept
    idxdkmppf = Y;

  end
end




