function kern = kRBF(theta)
    kern.theta = theta;
    kern.f   = @(x,y)     kernel(theta, x, y);
    kern.df1 = @(A,K,x,y) ddkernel(theta, A, K, x, y);
    kern.dfsym = @(A,K,x) ddkernelsym(theta, A, K, x);
    kern.dt{1}  = @(K,x,y)  dt1kernel(theta, K, x, y);
    kern.dt{2}  = @(K,x,y)  dt2kernel(theta, K, x, y);
    kern.fgrad = @(x,y) gradkernel(theta, x, y);
end

% compute kernel over a multi-input pair
function K = kernel(theta, X, Y)
%EARLIER ONE
%     N1 = size(X,2);
%     N2 = size(Y,2);
%     K = theta(1)*exp( -distEucSq(X,Y) / (2*theta(2)^2) )
%     K = theta(2)*exp( -distEucSq(X,Y) / (2*theta(1)^2) )
%all other parts where distEucSq is used is not changed as only kernel.f is
%used for evaluation

%taken from spgp_pred.m from pI code
    X=X';
    Y=Y';
    [n1,dim] = size(X); n2 = size(Y,1);
%     b = exp(theta(1:end-1)); c = exp(theta(end));
    b = theta(1:end-1); c = theta(end);
    c = 1;
    
    X = X.*repmat(sqrt(b)',n1,1);
    Y = Y.*repmat(sqrt(b)',n2,1);
    
    K = -2*X*Y' + repmat(sum(Y.*Y,2)',n1,1) + repmat(sum(X.*X,2),1,n2);
    K = c*exp(-0.5*K);
    
end

function K = gradkernel(theta, X, Y)
    N1 = size(X,2);
    N2 = size(Y,2);
    % K = theta(1)*exp( -distEucSq(X,Y)*0.01 / (2) );

    K = theta(1)*(distEucSq(X,Y).*exp( -distEucSq(X,Y) / (2*theta(2)^2) ));
end

% compute the directional derivative of the kernel matrix along A
% conceptually:
%
% dX(i,j) = <A, dK/dX_ij>
%
% but since only the jth row of K depends on the jth column of X1,
%
% <A, dK/dX_ij> => <A.j, dK.j/dX_ij>
%
% and if we can express the derivative of the jth row with respect to
%  the jth column, then:
%
% dX(:,j) = [dK.j/dX_j]*A.j'
%
function dX = ddkernel(theta, A, K, X, Y)
    D = size(X,1);
    [N1 N2] = size(K);
    dX = zeros(size(X));
    for j = 1:N1
        %       dK_jk/dX_j       = (1/sig^2)*K(j,k)*(Y_k - X_j)
        %   =>  dK.j /dX_j       = (1/sig^2)*bsxfun(@minus, Y, X_j)*diag(K.j)
        %   => [dK.j /dX_j]*A.j' = (1/sig^2)*bsxfun(@minus, Y, X_j)*(K.j .* A.j)'
        dX(:,j) = (theta(1)/theta(2)^2)*bsxfun(@minus,Y,X(:,j))*(K(j,:).*A(j,:))';
    end
end

function dX = ddkernelsym(theta, A, K, X)
    dX = zeros(size(X));
    for j=1:size(X,2)
        dX(:,j) = 2*(theta(1)/theta(2)^2)*bsxfun(@minus,X,X(:,j))*(K(j,:).*A(j,:))';
    end
end

% http://www.cs.columbia.edu/~mmerler/project/code/pdist2.m
function D = distEucSq( X, Y )
    m = size(X,2); n = size(Y,2);
    XX = sum(X'.*X',2);
    YY = sum(Y.*Y,1);
    D = XX(:,ones(1,n)) + YY(ones(1,m),:) - 2*X'*Y;
end

function dt1 = dt1kernel(theta, K, X, Y)
    dt1 = 1/theta(1) * K;
end

function dt2 = dt2kernel(theta, K, X, Y)
    dt2 = 1/theta(2)^3 * (K.*distEucSq(X,Y));
end
