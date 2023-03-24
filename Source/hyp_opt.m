function [theta, noise_prior] = hyp_opt(hyp_trainX,hyp_trainY)

% Initializing Gaussian kernel bandwidth parameter. Here we use ARD
% (Automatic Relevance Determination), i.e., we have separate theta for
% every dimension of the input vector
% We have dim+2 bcoz we also consider the amplitude and variance of noise.
% Thus causing two extra parameters.


global hyp_xtrain;
global y0_hyp_ytrain_pog;

hyp_xtrain = hyp_trainX - mean(hyp_trainX);

%center/standardize the data
y0_hyp_ytrain_pog=hyp_trainY-mean(hyp_trainY);
% y0_hyp_ytrain_pog=(hyp_trainY-mean(hyp_trainY,1))./std(hyp_trainY,0,1);

dim = size(hyp_trainX, 2);
% theta= rand(dim+2,1);
theta = [0.151899863776313;0.152753913195153;0.167993342993275;0.2739];

% Initialization of various variables required for hyperparameter
% optimisation
A = [];
b = [];
Aeq = [];
beq = [];
lb= 1e-8*ones(dim+2,1);
ub=[];
nonlcon = [];

% Hyperparameter Optimisation using inbuilt matlab function "fmincon"
options = optimset('PlotFcns',@optimplotfval);
%options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton');
% theta=fminsearch(@neglog_marginallikelihood,theta,options);
theta=fmincon(@neglog_marginallikelihood,theta,A,b,Aeq,beq,lb,ub,nonlcon,options);

noise_prior=sqrt(theta(end));
theta=theta(1:end-1);


end