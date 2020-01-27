% Jonathan Lu
% AMATH 482
% HW3 - Identifying Music
% 2/16/17
close all; clc
%% load in the files, convert to spectrogram .mat files
%for k = 1:5
%    audio_file = strcat('rap',sprintf('%d',k));
%    processAudioFile(audio_file, 60, 5);
% end
% for k = 1:5
%    audio_file = strcat('Classical',sprintf('%d',k));
%    processAudioFile(audio_file, 60, 5);
% end
% for k = 1:5
%    audio_file = strcat('FH',sprintf('%d',k));
%    processAudioFile(audio_file, 60, 5);
% end
% 
% adding the test data
% processAudioFile('raptest',60,5);
% processAudioFile('Classicaltest',60,5);
% processAudioFile('FHtest',60,5);
% 
% turn .mat files into grayscale, double precision, reshape to vector,
% add to data matrix for processing
A = [];
w = 875; % width of .mat 
h = 656; % height

for j = 1:5
    for k = 1:5
        matName = strcat('rap',sprintf('%d',j),'_',sprintf('%d',k));
        R = reshape(double(rgb2gray(eval(matName))), [], 1);
        A = [A R];
    end
end

for j = 1:5
    for k = 1:5
        matName = strcat('Classical',sprintf('%d',j),'_',sprintf('%d',k));
        R = reshape(double(rgb2gray(eval(matName))), [], 1);
        A = [A R];
    end
end

for j = 1:5
    for k = 1:5
        matName = strcat('FH',sprintf('%d',j),'_',sprintf('%d',k));
        R = reshape(double(rgb2gray(eval(matName))), [], 1);
        A = [A R];
    end
end

%test data
for k = 1:5
    matName = strcat('raptest_',sprintf('%d',k));
    R = reshape(double(rgb2gray(eval(matName))), [], 1);
    A = [A R];
end
for k = 1:5
    matName = strcat('Classicaltest_',sprintf('%d',k));
    R = reshape(double(rgb2gray(eval(matName))), [], 1);
    A = [A R];
end
for k = 1:5
    matName = strcat('FHtest_',sprintf('%d',k));
    R = reshape(double(rgb2gray(eval(matName))), [], 1);
    A = [A R];
end

[U, S, V] = svd(A', 'econ');
save('dataMat')

%load('dataMat');
%% singular value modes
figure(1)
plot(diag(S), 'ko', 'Linewidth', 2) %one huge mode (average spectrogram),
xlabel('\sigma_x')
ylabel('Magnitude')
title('Magnitude of \sigma_x')

% for j = 1:4  %look at first 4 principal components.
%    subplot(2,2,j)
%    ef = reshape(V(:,j),875,656);
%    pcolor(ef),axis off, shading interp, colormap(hot)
% end

%% look at projection of each artist onto dominant modes (2-4)
figure(2)
plot3(U(1:25,2),U(1:25,3),U(1:25,4), 'ko') %rap
hold on
plot3(U(26:50,2),U(26:50,3),U(26:50,4), 'ro') %classical
plot3(U(51:end,2),U(51:end,3),U(51:end,4), 'go') %future house
title('Heterogenous Data')
legend('Tupac', 'Beethoven', 'Jacob Tillberg', 'location' ,'best')
%data is mixed, but potential to separate the data

%% cross validation
% random permutation indices for choosing spectrograms
% q1 = randperm(25);
% q2 = randperm(25);
% q3 = randperm(25);
% 
xRap = U(1:25, 2:4);
xClassical = U(26:50, 2:4);
xFH = U(51:75, 2:4);
% 
% %training set (80% of original)
% x_train = [xRap(q1(1:20), :); xClassical(q2(1:20), :); xFH(q3(1:20), :)];
% %testing set (remaining 20%)
% x_test = [xRap(q1(21:end), :); xClassical(q2(21:end), :); xFH(q3(21:end), :)];
% 
% % predefined labels: where 1 represents rap, 2 classical, 3 future house
% c_train = [ones(20,1); 2*ones(20,1); 3*ones(20,1)];

%% knn-search (2)
X = [xRap; xClassical; xFH]; %data
C = [ones(25,1); 2*ones(25,1); 3*ones(25,1)]; %labels
md1 = fitcknn(X, C, 'NumNeighbors', 4);
md1_cv = crossval(md1); %10-fold training algorithm for cross validation
kloss = kfoldLoss(md1_cv); %see accuracy of program
X_test = [U(76:80, 2:4);U(81:85, 2:4);U(86:90, 2:4)];
pred = predict(md1, X_test);
bar(1:5, pred(1:5),'k')
hold on
bar(6:10, pred(6:10), 'r')
bar(11:15, pred(11:15), 'g')
set(gca, 'Xtick', 1:15)
title('Song predictions')
xlabel('Sample Number')
ylabel('Label Value')
legend('Chopin', 'Beethoven', 'Mozart', 'location' ,'northwest')

%% Naive-Bayes supervised classification (1)
nb = fitcnb(X, C);
cval = crossval(nb);
kloss = kfoldLoss(cval); %0.8
pred = nb.predict(X_test);
bar(pred)
bar(1:5, pred(1:5),'k')
hold on
bar(6:10, pred(6:10), 'r')
bar(11:15, pred(11:15), 'g')
set(gca, 'Xtick', 1:15)
title('Song predictions')
xlabel('Sample Number')
ylabel('Label Value')
legend('Tupac', 'Beethoven', 'Jacob Tillberg', 'location' ,'northwest')

%% LDA (Linear Discriminant Analysis) (3)
lda = fitcdiscr(X,C);
cval = crossval(lda);
kloss = kfoldLoss(cval); %0.7
pred = lda.predict(X_test);
bar(pred)
bar(1:5, pred(1:5),'k')
hold on
bar(6:10, pred(6:10), 'r')
bar(11:15, pred(11:15), 'g')
set(gca, 'Xtick', 1:15)
title('Song predictions')
xlabel('Sample Number')
ylabel('Label Value')
legend('Rap', 'Classical', 'Future House', 'location' ,'northwest')

%% SVM (Support Vector Machines)
% Note: can train with very little data (just need support vectors) -
% different from many other algorithms
% can be used for separated data
% svm = fitcsvm(x_train, c_train);
% pred = classificationsvm(svm, x_test);
% bar(pred)
