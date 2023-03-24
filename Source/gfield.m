function Z = gfield(X)

mu = [[2 2];[-3 -3];[5 -4];[-6 6]];
cov1 = [5 0; 0 3];
cov2 = [4 0; 0 4];      
cov3 = [9 0; 0 9];
cov4 = [1.5 0; 0 1.5];

z1 = 2*pi*sqrt(det(cov1))*mvnpdf(X,mu(1,:),cov1); %峰值1的2维高斯分布
z2 = 0.7*2*pi*sqrt(det(cov2))*mvnpdf(X,mu(2,:),cov2); %峰值0.7的2维高斯分布
z3 = 0.45*2*pi*sqrt(det(cov3))*mvnpdf(X,mu(3,:),cov3); %峰值0.7的2维高斯分布
z4 = 0.8*2*pi*sqrt(det(cov4))*mvnpdf(X,mu(4,:),cov4); %峰值0.7的2维高斯分布

Z = z1+z2+z3+z4;



end