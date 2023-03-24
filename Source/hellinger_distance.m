function [hellinger,bhattacharyya] = hellinger_distance(mu1,mu2,Sigma1,Sigma2)
%computes the Hellinger distance between two multivariate gaussians with
%means mu1, mu2 and covariances Sigma1, Sigma2
u=mu1-mu2;
detsss=(det(Sigma1)^(1/4)*det(Sigma2)^(1/4))/(det(.5*Sigma1+.5*Sigma2)^.5);

if isnan(detsss)
    detsss = 1;
end

bhattacharyya=detsss*exp(-(1/8)*u'/(.5*Sigma1+.5*Sigma2)*u);
hellinger=sqrt(1-bhattacharyya);
end

