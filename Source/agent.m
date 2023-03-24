classdef agent
    properties
        dataX
        dataY
        kernel
        noise_prior
        eps
        D
        Dy
        mu
        sigma
        Dist_mu
        Dist_sigma
        Dist_sinv
        theta
        Fused_mu
        Fused_sigma
        rsigma
        rs
        rtheta
        distributd_flag
        time
    end
    methods
        function obj = agent(dataX,dataY,kernel,noise_prior,eps,nTest)
            obj.dataX = dataX;
            obj.dataY = dataY;
            obj.kernel = kernel;
            obj.noise_prior = noise_prior;
            obj.eps = eps;
            obj.D = [];
            obj.Dy = [];
            obj.time = 1;
            
            % for GPR prediction
            obj.sigma = zeros(nTest,1);
            obj.mu = zeros(nTest,1);
            obj.Fused_sigma = zeros(nTest,1);
            obj.Fused_mu = zeros(nTest,1);

            % for dynamic average consensus
            obj.theta = zeros(nTest,1);
            obj.Dist_sigma = zeros(nTest,1);
            obj.Dist_sinv = zeros(nTest,1);
            obj.Dist_mu = zeros(nTest,1);
            obj.rsigma = zeros(nTest,1);
            obj.rs = zeros(nTest,1);
            obj.rtheta = zeros(nTest,1);
            obj.distributd_flag = 0;

        end

        function obj = Train(obj,newX,newY,testX)

            Dtmp = [obj.D newX];             
            Dytmp = [obj.Dy newY];

            % design a descreasing function f for eps -> 0
            T = 800;
            f = 2/(1+exp(5*obj.time/T)); 

%             f = 1;
            eps_h = obj.eps*f;%*obj.time/200; 
            
            DataSize = size(Dtmp,2);
            Nlimit = 30;

            if  DataSize < Nlimit
                obj.D = Dtmp;
                obj.Dy = Dytmp;
            else
                idxdkmppf = Compress_FB(Dtmp,Dytmp,obj.kernel,testX',eps_h,obj.noise_prior,obj.Fused_mu,obj.Fused_sigma);
                obj.D = Dtmp(:,idxdkmppf);
                obj.Dy = Dytmp(idxdkmppf);
            end
        end


        function obj = Predict(obj,testX)
            % GPR Prediction
            % Reference: Gaussian Processes for Machine Learning, Williams

            nTest = size(testX,1);
            mu_tmp = zeros(nTest,1);   
            sigma_tmp = zeros(nTest,1);

            y = obj.Dy;        
            KDD = obj.kernel.f(obj.D,obj.D);

            for i = 1:nTest
                xtest_pt = testX(i,:);
                KxtestD = obj.kernel.f(obj.D, xtest_pt');
                mu_tmp(i)= KxtestD'/(KDD + obj.noise_prior^2*eye(size(KDD)))*y';
                sigma_tmp(i) = obj.kernel.f(xtest_pt', xtest_pt') - KxtestD'/(KDD + obj.noise_prior^2*eye(size(KDD)))*KxtestD + obj.noise_prior^2;
            end    
            obj.sigma = sigma_tmp;
            obj.mu = mu_tmp;
        end
        

        function obj = fused_GP(obj)
            % Fuse two GPR distributions according to its covariance 
            % Reference: Communication-aware Distributed Gaussian Process Regression 
            % Algorithms for Real-time Machine Learning, 2020ACC
            
            obj.Fused_sigma = obj.Dist_sigma;
%             obj.Fused_mu = obj.Dist_mu;

            N = size(obj.sigma,1);
            flag = zeros(N,1);
            for i = 1:N
                if obj.sigma(i,1) < obj.Dist_sigma(i,1)
                    flag(i,1) = 1;
                    obj.Fused_sigma(i,1) = obj.sigma(i,1);
                end
            end
            
            obj.Fused_mu = obj.mu.*flag + obj.Dist_mu.*(1-flag);
        end
        
     

    end


end