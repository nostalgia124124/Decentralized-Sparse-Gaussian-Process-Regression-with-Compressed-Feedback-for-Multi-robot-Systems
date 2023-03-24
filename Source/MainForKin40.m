% main for kin40 datasets
% if change dataset, Nlimit, epsilion_t and steplength of DAC should be
% changed

clear;
close all;
warning('off');

%-------- kin40 datasets  -----------
load('data_kin40.mat');
trainX = data.xtrain;
trainY = data.ytrain;
testX = data.xtest;
testY = data.ytest;


Nt = size(trainX,1);
nTest = size(testX,1);
hyp_trainsize = 0;

rand_init_train = randperm(Nt,hyp_trainsize);
train_it = setdiff(1:Nt, rand_init_train);
hyp_train_it = setdiff(1:Nt, train_it);
TX = trainX;
TY = trainY;

% ------------para for kin40------------
Keps = 4*10^(-2);


% [theta, noise_prior] = hyp_opt(TX(1:hyp_trainsize,:),TY(1:hyp_trainsize,:));

theta = [0.038;0.954;0.742;0.937;0.513;0.241;0.260;0.759;0.993];
noise_prior = 0.2739;
kernel = kRBF(theta);

% TX = TX(train_it, :);
% TY = TY(train_it, :);

MeanY = mean(TY);
TY_center = TY - MeanY; 

trainN = Nt - hyp_trainsize;
robot_num = 5;
Sample_gap = trainN/robot_num;


Robot1 = agent(TX(1:Sample_gap,:)',TY_center(1:Sample_gap,:)',kernel,noise_prior,Keps,nTest);
Robot2 = agent(TX(Sample_gap+1:Sample_gap*2,:)',TY_center(Sample_gap+1:Sample_gap*2,:)',kernel,noise_prior,Keps,nTest);
Robot3 = agent(TX(Sample_gap*2+1:Sample_gap*3,:)',TY_center(Sample_gap*2+1:Sample_gap*3,:)',kernel,noise_prior,Keps,nTest);
Robot4 = agent(TX(Sample_gap*3+1:Sample_gap*4,:)',TY_center(Sample_gap*3+1:Sample_gap*4,:)',kernel,noise_prior,Keps,nTest);
Robot5 = agent(TX(Sample_gap*4+1:Sample_gap*5,:)',TY_center(Sample_gap*4+1:Sample_gap*5,:)',kernel,noise_prior,Keps,nTest);

RobotArray = [Robot1,Robot2,Robot3,Robot4,Robot5];
Sample_Nt = 5; 


% Store history error
FusedError = zeros(robot_num,Sample_gap);
LocalError = zeros(robot_num,Sample_gap);


for it = 1:Sample_gap

    % Each robot train GPR and predict locally
    for i = 1:robot_num
        RobotArray(1,i).time = it;
        RobotArray(1,i) = RobotArray(1,i).Train(RobotArray(1,i).dataX(:,it),RobotArray(1,i).dataY(:,it),testX);
        DataSize(i,it) = size(RobotArray(1,i).D,2);
        RobotArray(1,i) = RobotArray(1,i).Predict(testX);
        RobotArray(1,i).mu = RobotArray(1,i).mu + MeanY; 

        LocalError(i,it) = norm(RobotArray(1,i).mu - testY)/size(testY,1);
    end
        

    fprintf('it:%d, DataSize:%d, Local mse: %.3d, ', it, sum(DataSize(:,it)), mean(LocalError(:,it)));

    % Communicate and achieves dynamic average consensus
    RobotArray = DAC(RobotArray);
    
    for i = 1:robot_num
        RobotArray(1,i) = RobotArray(1,i).fused_GP();
        FusedError(i,it) = mean(norm(RobotArray(1,i).Fused_mu - testY)/size(testY,1));
    end
    FusedMeanError(1,it) = mean(FusedError(:,it));
    
    fprintf('Distributed mse: %.3d \n', FusedMeanError(1,it));

end


%%
figure;
plot(mean(FusedError,1));