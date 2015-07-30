%trainSceneLabels - script used to train eight different types of scene

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set up the environment to run SVM on data
%clear; clc; close all
%load 'genfeatures';
categories = {'Sky','Tree','Road','Grass','Water','Bldg','Mtn','Fground'};
Nclasses = length(categories);
foldidx = 4;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extract the global features for all image superpixels
% We assume the following variables are in Matlab workspace:          
%   features             715x1             cell                
%   image_data             1x715           struct              
%   imsegs                 1x715           struct              
%   keys              396133x2             double 
%   label_color_map        8x3             double              
%   labels               715x1             cell                
%   test_idx               1x5             cell                
%   train_idx              1x5             cell                
disp('Extracting global spix information')
tmp = features{1};
Nfeatures = size(tmp,2);
Nspix = size(keys, 1);

% Loop: Extract features for all superpixels in database
F = zeros([Nspix Nfeatures]);
C = zeros([Nspix 1]);
for n = 1:Nspix
  img_id = keys(n, 1); 
  sp_id = keys(n, 2); 
  F(n,:) = features{img_id}(sp_id,:);
  C(n) =  labels{img_id}(sp_id);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DO SVM TRAINING OR YOUR OWN CLASSIFIER TRAINING HERE....
% train one versus all and store all the models (1 class versus others)
disp('Setting up the spix for the training dataset')
trainx = train_idx{foldidx}; %NOTE to change this for next validation round
train_ids = zeros(size(F,1),1);
counter=0;
for n = 1:length(trainx)
  img_id = trainx(n);
  keys_img = keys(:,1);
  tmpids = find(keys_img == img_id); 
  train_ids(counter+1:counter+size(tmpids,1))= tmpids;
  counter=counter+size(tmpids,1); 
end
train_ids = train_ids(1:counter);
Ftrain = F(train_ids,:); %Use this as your training data
Ctrain = C(train_ids); %Use this as the labels for your training data

%
len = length(train_ids);
randlist = randperm(len);
Ftrain = Ftrain(randlist (1:5000),:);
Ctrain = Ctrain(randlist (1:5000),:);

M = 44;                            % Number of features
H = (2./3).* 45;                    % Hidden Layers
K = 8;                             % classes
N = size(Ftrain,1);             % Total number of inputs
cntr = 0; 
num_steps = 0; 

n1 = 0.00001;
n2 = 0.0000001;
E_cross = inf;                      % The cross entropy value is initially INF
Error = inf;
E_result = [];

Y = zeros(N,K);

%% Model the variables for further operations

% Output Variable T must be created
T_train = zeros(size(Ctrain,1),K);
for i = 1:size(Ctrain,1)
    for j = 1:8
        if Ctrain(i) == j
            T_train(i,j) = 1;
            break;
        end 
    end
end


PHI = Ftrain;
PHI = [ones(size(PHI,1),1) PHI(:,1:44)];

W_l1 = rand(M+1,H)-0.5;
W_l2 = rand(H,K)-0.5;

W_n1 = W_l1;
W_n2 = W_l2;

W_u1 = zeros(M+1,H);
W_u2 = zeros(H,K);

while (num_steps <= 4000 && ((Error > 0 )&& ~isnan(E_cross)))
    % Calculate for the hidden layer 1
    Aj = PHI*(W_l1);
    sigma_Aj = 1./(1+exp(-Aj));
    % Calculate for the hidden layer 2
    Ak = sigma_Aj*(W_l2);
    expAk = exp(Ak);
    sumAk = sum(expAk,2);
    
    %Using the above Calculate the output function 
    for i = 1:N
        for j = 1:K
            Y(i,j) = expAk(i,j)./sumAk(i,1);
        end
    end
    
    %Calculate the Error function (cross Entropy)
    
    E_cross = -sum(sum(T_train.*log(Y)));
    num_steps = num_steps + 1;
    E_result = [E_result;E_cross];
    fprintf('Step number: %d E_cross %d\n',num_steps,E_cross); 
    fprintf('Cross_Entropy: %d Step %d\n',E_cross,num_steps);
    
    Error_k = Y-T_train;
    Error_j = (sigma_Aj.*(1-sigma_Aj)).*(Error_k*W_l2');

    delEk = PHI'*Error_j;
    delEj = sigma_Aj'*Error_k;
    
    % Performing the eta correction
    % Estimating the step size
    
    if (Error <= E_cross)
        cntr = 0; 
        n1 = n1+ 0.00001;
        n2 = n2+ 0.0000001;
    else
        cntr = cntr+1;
        if cntr > 3
            cntr = 0;
            n1 = n1 + 0.0001;
            n2 = n2 + 0.000001;
        end
        W_u1 = W_n1;
        W_u2 = W_n2;
        Error = E_cross;
    end
    
    W_n1 = W_l1-(n1.*delEk);
    W_n2 = W_l2-(n2.*delEj);
    
    W_l1 = W_n1;
    W_l2 = W_n2;
end
W1 = W_u1;
W2 = W_u2;

% disp('Running the SVM fit model process')
% netall = cell(Nclasses, 1); 
% for c = 1:Nclasses
%     SVMModel = fitcsvm(Ftrain, 2*(Ctrain==c)-1,'ClassNames',[1 -1],...
%         'KernelFunction','rbf','Standardize',true);
%     netall{c} = SVMModel;
% end

disp('Completed training ***remember to save workspace***')
