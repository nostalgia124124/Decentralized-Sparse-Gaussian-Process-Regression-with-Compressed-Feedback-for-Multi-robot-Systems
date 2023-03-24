%  main for scalar fields case

clear;
close all;
warning('off');

%-------- generate the scalar fields  -----------
Nt = 400; 
robot_num = 5;
iter = 1:Nt;
Source = [5 5;5 -5;-5 -5;-5 5;0 0];
rx = zeros(5,Nt);
ry = zeros(5,Nt);
TX = zeros(5,Nt,2);
TZ = zeros(5,Nt);

for i = 1:robot_num
    rx(i,:) = Source(i,1)+(5.*iter/Nt).*sin(3.2*2*pi*iter/Nt);
    ry(i,:) = Source(i,2)+(5.*iter/Nt).*cos(3.2*2*pi*iter/Nt);
    TX(i,:,:) = [rx(i,:)' ry(i,:)'];
    TZ(i,:) = gfield(reshape(TX(i,:,:),Nt,2));
end

trainX = reshape(TX,Nt*robot_num,2);
trainY = reshape(TZ,Nt*robot_num,1);
Nt = Nt*robot_num;

Nm = 20;
[testx1, testx2] = meshgrid(linspace(-10,10,Nm));
testX = [testx1(:) testx2(:)];
testY = gfield(testX);
nTest = size(testX,1);

TX = trainX;
TY = trainY;

% ------------para for field------------
Keps = 3*10^(-6);
%

hyp_trainsize = 0;
% hyp_trainsize = 500;
% [theta, noise_prior] = hyp_opt(TX(1:hyp_trainsize,:),TY(1:hyp_trainsize,:));

theta = [0.151899863776313;0.152753913195153;0.167993342993275];
noise_prior = 0.2739;
kernel = kRBF(theta);

train_it = 1:(Nt-hyp_trainsize);
TX = TX(train_it, :);
TY = TY(train_it, :);

MeanZ = mean(TZ,'all');
TY_center = TZ - MeanZ; 

trainN = Nt - hyp_trainsize;
robot_num = 5;
Sample_gap = trainN/robot_num;

Robot1 = agent([rx(1,:);ry(1,:)],TY_center(1,:),kernel,noise_prior,Keps,nTest);
Robot2 = agent([rx(2,:);ry(2,:)],TY_center(2,:),kernel,noise_prior,Keps,nTest);
Robot3 = agent([rx(3,:);ry(3,:)],TY_center(3,:),kernel,noise_prior,Keps,nTest);
Robot4 = agent([rx(4,:);ry(4,:)],TY_center(4,:),kernel,noise_prior,Keps,nTest);
Robot5 = agent([rx(5,:);ry(5,:)],TY_center(5,:),kernel,noise_prior,Keps,nTest);

RobotArray = [Robot1,Robot2,Robot3,Robot4,Robot5];


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
        RobotArray(1,i).mu = RobotArray(1,i).mu + MeanZ; 

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

%% plot for fields
close all;
colorlist = [[1,0,0];[1,1,0];[1,0.38,0];[0.13,0.55,0.13];[0.25,0.4,0.88]];

% Local GPR visualization
for i = 1:robot_num
    figure;
    surf(testx1,testx2,reshape(RobotArray(1,i).mu,Nm,Nm),'EdgeColor','none');hold on;
    xlabel('X asix');
    ylabel('Y asix');
    zlim([-0.1,1]);
    view([28.9,36.8]);
end

% Distributed GPR visualization
for i = 1:robot_num
    figure;
    surf(testx1,testx2,reshape(RobotArray(1,i).Dist_mu,Nm,Nm),'EdgeColor','none');hold on;
    xlabel('X asix');
    ylabel('Y asix');
    zlim([-0.1,1]);
    view([28.9,36.8]);
end

