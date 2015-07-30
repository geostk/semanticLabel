%testSceneLabels - script used to test after SVM training is completed

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set up the environment to run SVM on data
%clear; clc; close all
%load 'svm_train_wspace';
categories = {'Sky','Tree','Road','Grass','Water','Bldg','Mtn','Fground'};
Nclasses = length(categories);
foldidx = 1;

%setup the features for testing
disp('Setting up the spix for the testing dataset')
test = test_idx{foldidx}; %NOTE to change this to match the training indexes
test_ids = zeros(size(F,1),1);
counter=0;
for n = 1:length(test)
  img_id = test(n);
  keys_img = keys(:,1);
  tmpids = find(keys_img == img_id);
  test_ids(counter+1:counter+size(tmpids,1))= tmpids;
  counter=counter+size(tmpids,1); 
end
test_ids = test_ids(1:counter);
Ftest = F(test_ids,:); %This as our testing data
Ctest = C(test_ids); %Use this as the labels for your training data

T_train = zeros(size(Ctest,1),K);
for i = 1:size(Ctrain,1)
    for j = 1:8
        if Ctrain(i) == j
            T_train(i,j) = 1;
            break;
        end 
    end
end

M = 44;                            % Number of features
K = 8;                             % classes
N = size(Ftest,1);                          % Total number of inputs
cntr = 0; 
answer = zeros(N,K);
%% Model the variables for further operations

% Create PHI values 
PHI = Ftest;
PHI = [ones(N,1) PHI(:,1:M)];     % Bias Value

Aj = PHI*(W1);
sigma_Aj = 1./(1+exp(-Aj));
fprintf('Size %d-%d W %d-%d\n',size(sigma_Aj,1),size(sigma_Aj,2),size(W2,1),size(W2,2));
Ak = sigma_Aj*(W2);


expAk = exp(Ak);
sumAk = sum(expAk,2);

expected_ans = zeros(N,K);
for i=1:N
    for j=1:K;
        expected_ans(i,j) = expAk(i,j)./sumAk(i,1);
    end
end

[~,n] = max(expected_ans,[],2);

for i= 1:N
    answer(i,n(i,1)) = 1;
end

miss = (sum(sum(abs(T_train-answer))))/2;
error = (miss/size(PHI,1)).*100;

fprintf('The error rate is %d\n',error);

[~,T_label] = max(expected_ans,[],2);
T_label = T_label-1;


y_final=T_label;




%{
% Loop: compute the maximal score for features
% disp('Using the SVM models to predict scores for test data')
scores = zeros(Nclasses, length(test_ids));
for c = 1:Nclasses
    disp(['Scores for class ' num2str(c)])
    [~,score] = predict(netall{c},Ftest);
    scores(c,:) = score(:,1);
end
whos scores netall Ftest

ctest_hat = zeros(length(test_ids), 1);
for k = 1:length(test_ids)
  [foo, ctest_hat(k)] = max(scores(:,k));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This section below is optional
% Plot performance
% Confusion matrix:
disp('Setting up the confusion matrix')
Cm = zeros(Nclasses, Nclasses);
for j = 1:Nclasses
    for i = 1:Nclasses
        % row i, col j is the percentage of images from class i that
        % were missclassified as class j.
        Cm(i,j) = 100*sum((C(test_ids)==i) .* (ctest_hat==j))/(0.0001+sum(C(test_ids)==i));
    end
end

figure
subplot(121)
imagesc(Cm); axis('square'); colorbar
subplot(122)
bar(diag(Cm))
title(mean(diag(Cm)))
axis('square')
%}