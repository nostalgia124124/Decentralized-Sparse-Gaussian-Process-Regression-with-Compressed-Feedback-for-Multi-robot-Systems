%DAC_avg.m
%
%DESCRIPTION:
%   implements dynamic average consensus for fusion of local GPR results
%   converge to the averaged results 
%
%INPUTS & OUTPUTS:
%   *RobotArray: the array of class agent
%
%Reference:
%   Discrete-time dynamic average consensusDiscrete-time dynamic average
%   consensus, 2010, Automatica

function RobotArray = DAC_avg(RobotArray)

    t = RobotArray(1,1).time;
% set the adjust Matrix, the number of robots is 5 
% constant
%     A = [0.4 0.2 0 0.2 0.2;
%          0.2 0.6 0.2 0 0;
%          0 0.2 0.6 0.2 0;
%          0.2 0 0.2 0.4 0.2;
%          0.2 0 0 0.2 0.6];

% time-varying

    A = [0.7 0.1 0 0 0.2;
        0.1 0.7 0.2 0 0;
        0 0.2 0.7 0.1 0;
        0 0 0.1 0.7 0.2;
        0.2 0 0 0.2 0.6] + [0.1*(-1)^t -0.1*(-1)^t 0 0 0;
        -0.1*(-1)^t 0.1*(-1)^t 0 0 0;
        0 0 -0.1*(-1)^t 0.1*(-1)^t 0;
        0 0 0.1*(-1)^t -0.1*(-1)^t 0;
        0 0 0 0 0];

    robot_num = 5;
    theta = []; 
    rtheta = [];
    oldtheta = [];
    
    sinv = [];
    rsinv = [];
    oldrsinv = [];
    
    eta = [];
    reta = [];
    oldreta = [];

    if RobotArray(1,1).distributd_flag == 0
        for i = 1:robot_num
            RobotArray(1,i).rtheta = RobotArray(1,i).mu;
            RobotArray(1,i).rsigma = RobotArray(1,i).sigma;
            RobotArray(1,i).rs = RobotArray(1,i).sigma.^(-1);
            
            RobotArray(1,i).Dist_mu = RobotArray(1,i).mu;
            RobotArray(1,i).Dist_sigma = RobotArray(1,i).sigma;
            RobotArray(1,i).Dist_sinv = RobotArray(1,i).sigma.^(-1);   

            RobotArray(1,i).distributd_flag = 1;
        end

    else

        for i = 1:robot_num
    
            theta = [theta RobotArray(1,i).Dist_mu];
            rtheta = [rtheta RobotArray(1,i).mu];
            oldtheta = [oldtheta RobotArray(1,i).rtheta];
    
            sinv = [sinv RobotArray(1,i).Dist_sinv];
            rsinv = [rsinv RobotArray(1,i).sigma.^(-1)];
            oldrsinv = [oldrsinv RobotArray(1,i).rs];
            
            eta = [eta RobotArray(1,i).Dist_sigma];
            reta = [reta RobotArray(1,i).sigma];
            oldreta = [oldreta RobotArray(1,i).rsigma];
        end
    
    
        % update DAC

        theta = theta*A + (rtheta - oldtheta)*0.7;
        sinv = sinv*A + (rsinv - oldrsinv)*0.7;
        eta = eta*A + (reta - oldreta)*0.7;
    
        for i = 1:robot_num
            RobotArray(1,i).rtheta = RobotArray(1,i).mu;
            RobotArray(1,i).rsigma = RobotArray(1,i).sigma;
            RobotArray(1,i).rs = RobotArray(1,i).sigma.^(-1);
    
            RobotArray(1,i).Dist_mu = theta(:,i);
            RobotArray(1,i).Dist_sigma = eta(:,i);
            RobotArray(1,i).Dist_sinv = sinv(:,i);   
        end

    end
end